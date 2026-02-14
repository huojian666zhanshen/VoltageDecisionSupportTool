# decision_service/app.py
from __future__ import annotations

import copy
import re
import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Literal

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from pypower.api import case14, case30

app = FastAPI(title="VoltageDecisionSupportTool", version="1.3.1")  # ✅ bump


# -------------------------
# HTTP client (reuse connections)
# -------------------------
_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.headers.update({"Content-Type": "application/json"})


# -------------------------
# Endpoint normalization
# -------------------------
def _normalize_pf_endpoint(url: str) -> str:
    """Allow passing base URL or full /run_pf URL."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("pf_endpoint must be a non-empty string")
    url = url.strip()
    if not re.match(r"^https?://", url):
        raise ValueError("pf_endpoint must start with http:// or https://")
    if url.endswith("/"):
        url = url[:-1]
    if not url.lower().endswith("/run_pf"):
        url = url + "/run_pf"
    return url


def _normalize_sc_endpoint(url: str) -> str:
    """Allow passing base URL or full /security_check URL."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("security_endpoint must be a non-empty string")
    url = url.strip()
    if not re.match(r"^https?://", url):
        raise ValueError("security_endpoint must start with http:// or https://")
    if url.endswith("/"):
        url = url[:-1]
    if not url.lower().endswith("/security_check"):
        url = url + "/security_check"
    return url


# -------------------------
# Coercion: accept dict/list OR stringified dict/json
# -------------------------
def _coerce_obj(x: Any, name: str) -> Dict[str, Any]:
    """
    Accept:
      - dict
      - JSON string: '{"vmin":0.95}'
      - Python dict string: "{'vmin':0.95}"
      - None -> {}
    """
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            v = json.loads(s)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
        try:
            v = ast.literal_eval(s)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    raise ValueError(f"{name} must be an object/dict (or JSON-string of an object)")


def _coerce_list(x: Any, name: str) -> List[Any]:
    """
    Accept:
      - list
      - JSON string of list: '[{"bus":3,"delta":-0.05}]'
      - Python list string: "[{'bus':3,'delta':-0.05}]"
      - None -> []
    """
    if isinstance(x, list):
        return x
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
    raise ValueError(f"{name} must be a list (or JSON-string of a list)")


def _model_dump(x: Any) -> Dict[str, Any]:
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    return dict(x)


# -------------------------
# Pydantic Schemas
# -------------------------
CaseID = Literal["case14", "case30"]
PFMethod = Literal["ac", "dc"]
Issue = Literal["voltage_low", "voltage_high"]


class Limits(BaseModel):
    vmin: float = Field(0.95, description="Voltage lower bound (p.u.)")
    vmax: float = Field(1.05, description="Voltage upper bound (p.u.)")
    thermal_limit_field: str = Field("rateA", description="Branch thermal limit field name if needed")


class RedispatchCtl(BaseModel):
    enabled: bool = False
    gen_buses: List[int] = Field(default_factory=list, description="Generator buses (MATPOWER bus numbers)")
    step_mw: float = 10.0
    max_steps: int = 5
    pg_min: Optional[float] = None
    pg_max: Optional[float] = None


class ShuntCtl(BaseModel):
    enabled: bool = False
    buses: List[int] = Field(default_factory=list, description="Shunt buses (MATPOWER bus numbers)")
    step_mvar: float = 5.0
    max_steps: int = 5
    capacitive_increases_bs: bool = True


class OLTCctl(BaseModel):
    enabled: bool = False
    branch_idx: List[int] = Field(default_factory=list, description="Branch row indices (0-based) for OLTC")
    step_tap: float = 0.0125
    max_steps: int = 5
    tap_min: float = 0.9
    tap_max: float = 1.1


class Controllables(BaseModel):
    redispatch: RedispatchCtl = Field(default_factory=RedispatchCtl)
    shunt: ShuntCtl = Field(default_factory=ShuntCtl)
    oltc: OLTCctl = Field(default_factory=OLTCctl)


class SearchCfg(BaseModel):
    max_candidates: int = 30
    topk: int = 5
    max_parallel_pf: int = 8
    pf_timeout_sec: int = 20
    max_combo_depth: int = 1

    # ΔV -> 负荷扰动经验映射（构造可复现仿真场景）
    q_sensitivity_mvar_per_pu: float = 200.0  # ΔV=-0.05 -> +10 Mvar
    p_sensitivity_mw_per_pu: float = 0.0      # 默认不注入P负荷


class DecisionRequest(BaseModel):
    pf_endpoint: HttpUrl = Field(..., description="Base URL or /run_pf URL of the Power Flow Tool")
    security_endpoint: Optional[HttpUrl] = Field(
        None, description="Base URL or /security_check URL of the Security Check Tool"
    )

    method: PFMethod = "ac"
    case_id: CaseID
    issue: Issue

    # ✅ Dify 友好：单个观测（object），避免 array items 丢失
    observation: Any = Field(
        default=None,
        description='Single observation, e.g. {"bus":3,"delta":-0.05}',
    )
    # ✅ 兼容旧客户端：list 观测
    observations: Any = Field(
        default_factory=list,
        description='Voltage observations, e.g. [{"bus":3,"delta":-0.05}]',
    )

    limits: Any = Field(default_factory=Limits)
    controllables: Any = Field(default_factory=Controllables)
    search: Any = Field(default_factory=SearchCfg)


class DecisionDebugRequest(BaseModel):
    pf_endpoint: HttpUrl
    security_endpoint: Optional[HttpUrl] = None
    method: PFMethod = "ac"
    case: Dict[str, Any]
    issue: Issue

    observation: Any = Field(default=None)
    observations: Any = Field(default_factory=list)

    limits: Any = Field(default_factory=Limits)
    controllables: Any = Field(default_factory=Controllables)
    search: Any = Field(default_factory=SearchCfg)


class TextObsRequest(BaseModel):
    text: str = Field(..., description="Natural language observation, e.g. '3号节点电压低了0.05 p.u.'")
    pf_endpoint: HttpUrl
    security_endpoint: Optional[HttpUrl] = None
    method: PFMethod = "ac"
    case_id: CaseID
    issue: Optional[Issue] = None
    limits: Any = Field(default_factory=Limits)
    controllables: Any = Field(default_factory=Controllables)
    search: Any = Field(default_factory=SearchCfg)


# -------------------------
# Case loading / normalization
# -------------------------
def _np_to_list(x: Any) -> Any:
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return x


def load_ieee_case_from_id(case_id: CaseID) -> Dict[str, Any]:
    if case_id == "case14":
        c = case14()
    elif case_id == "case30":
        c = case30()
    else:
        raise ValueError(f"Unsupported case_id: {case_id}")

    c2: Dict[str, Any] = {}
    for k, v in c.items():
        c2[k] = _np_to_list(v)
    c2["baseMVA"] = float(c2.get("baseMVA", 100.0))
    return c2


def normalize_matpower_case(case_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(case_obj, dict):
        raise ValueError("case must be a dict")
    c2: Dict[str, Any] = {}
    for k, v in case_obj.items():
        c2[k] = _np_to_list(v)
    c2["baseMVA"] = float(c2.get("baseMVA", 100.0))

    for key in ("bus", "gen", "branch"):
        mat = c2.get(key)
        if mat is None:
            raise ValueError(f"case missing '{key}'")
        if not isinstance(mat, list):
            raise ValueError(f"case '{key}' must be a list")
        if len(mat) > 0 and not isinstance(mat[0], list):
            raise ValueError(f"case '{key}' must be list of rows (list[list])")
    return c2


# -------------------------
# Candidate generation (MVP)
# -------------------------
def generate_candidates(issue: Issue, ctl: Controllables, max_cands: int) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []

    # 1) OLTC
    if ctl.oltc.enabled and ctl.oltc.branch_idx:
        direction = +1 if issue == "voltage_low" else -1
        for br in ctl.oltc.branch_idx:
            for s in range(1, ctl.oltc.max_steps + 1):
                cands.append({
                    "type": "oltc",
                    "branch_idx": int(br),
                    "delta_tap": float(direction * s * ctl.oltc.step_tap),
                    "steps": int(direction * s),
                })

    # 2) Shunt
    if ctl.shunt.enabled and ctl.shunt.buses:
        direction = +1 if issue == "voltage_low" else -1
        for b in ctl.shunt.buses:
            for s in range(1, ctl.shunt.max_steps + 1):
                cands.append({
                    "type": "shunt",
                    "bus": int(b),
                    "delta_mvar": float(direction * s * ctl.shunt.step_mvar),
                    "steps": int(direction * s),
                })

    # 3) Redispatch
    if ctl.redispatch.enabled and ctl.redispatch.gen_buses:
        direction = +1 if issue == "voltage_low" else -1
        for gb in ctl.redispatch.gen_buses:
            for s in range(1, ctl.redispatch.max_steps + 1):
                cands.append({
                    "type": "redispatch",
                    "gen_bus": int(gb),
                    "delta_mw": float(direction * s * ctl.redispatch.step_mw),
                    "steps": int(direction * s),
                })

    return cands[:max_cands]


# -------------------------
# Apply action to MATPOWER case (JSON form)
# -------------------------
BUS_I = 0
BUS_PD = 2
BUS_QD = 3
BUS_BS = 5
BUS_VM = 7

GEN_BUS = 0
GEN_PG = 1

BR_TAP = 8


def _find_bus_row(bus_mat: List[List[float]], bus_id: int) -> Optional[int]:
    for i, row in enumerate(bus_mat):
        if int(row[BUS_I]) == int(bus_id):
            return i
    return None


def apply_action(case: Dict[str, Any], action: Dict[str, Any], ctl: Controllables) -> Dict[str, Any]:
    c = copy.deepcopy(case)
    baseMVA = float(c.get("baseMVA", 100.0))
    bus = c["bus"]
    gen = c["gen"]
    branch = c["branch"]

    atype = action.get("type")

    if atype == "oltc":
        br = int(action["branch_idx"])
        if 0 <= br < len(branch):
            tap = float(branch[br][BR_TAP])
            if tap == 0.0:
                tap = 1.0
            tap += float(action["delta_tap"])
            tap = max(float(ctl.oltc.tap_min), min(float(ctl.oltc.tap_max), tap))
            branch[br][BR_TAP] = tap

    elif atype == "shunt":
        b = int(action["bus"])
        ridx = _find_bus_row(bus, b)
        if ridx is not None:
            delta_mvar = float(action["delta_mvar"])
            sign = +1.0 if ctl.shunt.capacitive_increases_bs else -1.0
            #bus[ridx][BUS_BS] = float(bus[ridx][BUS_BS]) + sign * (delta_mvar / baseMVA)
            bus[ridx][BUS_BS] = float(bus[ridx][BUS_BS]) + sign * (delta_mvar)

    elif atype == "redispatch":
        gb = int(action["gen_bus"])
        delta_mw = float(action["delta_mw"])
        k = None
        for i, row in enumerate(gen):
            if int(row[GEN_BUS]) == gb:
                k = i
                break
        if k is not None:
            new_pg = float(gen[k][GEN_PG]) + delta_mw
            if ctl.redispatch.pg_min is not None:
                new_pg = max(float(ctl.redispatch.pg_min), new_pg)
            if ctl.redispatch.pg_max is not None:
                new_pg = min(float(ctl.redispatch.pg_max), new_pg)
            gen[k][GEN_PG] = new_pg

    return c


# -------------------------
# NEW: observations -> perturb loads to create reproducible scenario
# -------------------------
def apply_voltage_observations_to_case(
    base_case: Dict[str, Any],
    observations: List[Dict[str, Any]],
    q_sensitivity_mvar_per_pu: float = 200.0,
    p_sensitivity_mw_per_pu: float = 0.0,
) -> Dict[str, Any]:
    """
    Turn (bus, deltaV) into a reproducible scenario by perturbing loads at the bus.
    NOTE:
      - We do NOT set BUS_VM directly (PF solves Vm).
      - Here we modify MATPOWER PD/QD (units MW/Mvar in standard MATPOWER tables).
    """
    c = copy.deepcopy(base_case)
    bus = c["bus"]

    for ob in observations:
        b = int(ob.get("bus"))
        dv = float(ob.get("delta"))
        ridx = _find_bus_row(bus, b)
        if ridx is None:
            continue

        if dv < 0:
            add_q_mvar = (-dv) * float(q_sensitivity_mvar_per_pu)
            bus[ridx][BUS_QD] = float(bus[ridx][BUS_QD]) + add_q_mvar

            if p_sensitivity_mw_per_pu and float(p_sensitivity_mw_per_pu) > 0:
                add_p_mw = (-dv) * float(p_sensitivity_mw_per_pu)
                bus[ridx][BUS_PD] = float(bus[ridx][BUS_PD]) + add_p_mw

        elif dv > 0:
            sub_q_mvar = (dv) * float(q_sensitivity_mvar_per_pu)
            bus[ridx][BUS_QD] = max(0.0, float(bus[ridx][BUS_QD]) - sub_q_mvar)

            if p_sensitivity_mw_per_pu and float(p_sensitivity_mw_per_pu) > 0:
                sub_p_mw = (dv) * float(p_sensitivity_mw_per_pu)
                bus[ridx][BUS_PD] = max(0.0, float(bus[ridx][BUS_PD]) - sub_p_mw)

    return c


# -------------------------
# Call Power Flow Tool (robust)
# -------------------------
def call_run_pf(pf_endpoint: str, method: PFMethod, case_payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    url = _normalize_pf_endpoint(pf_endpoint)
    payload = {"method": method, "case": case_payload}

    try:
        r = _SESSION.post(url, json=payload, timeout=timeout_sec)
    except requests.Timeout:
        raise RuntimeError("timeout")
    except requests.RequestException as e:
        raise RuntimeError(str(e))

    if r.status_code >= 400:
        txt = (r.text or "")[:800]
        raise RuntimeError(f"http {r.status_code}: {txt}")

    try:
        return r.json()
    except Exception:
        txt = (r.text or "")[:800]
        raise RuntimeError(f"non-json response: {txt}")


# -------------------------
# Call Security Check Tool
# -------------------------
DEFAULT_SECURITY_ENDPOINT = "https://sec.gridgpt.dev/security_check"  # ✅ 改成你实际域名


def call_security_check(
    security_endpoint: str,
    pf_result: Dict[str, Any],
    limits: Limits,
    timeout_sec: int
) -> Dict[str, Any]:
    url = _normalize_sc_endpoint(security_endpoint)
    payload = {"pf": pf_result, "limits": _model_dump(limits)}

    try:
        r = _SESSION.post(url, json=payload, timeout=timeout_sec)
    except requests.Timeout:
        raise RuntimeError("timeout")
    except requests.RequestException as e:
        raise RuntimeError(str(e))

    if r.status_code >= 400:
        txt = (r.text or "")[:800]
        raise RuntimeError(f"http {r.status_code}: {txt}")

    try:
        return r.json()
    except Exception:
        txt = (r.text or "")[:800]
        raise RuntimeError(f"non-json response: {txt}")


# -------------------------
# Parse helpers (PF outputs)
# -------------------------
def extract_vm_list(pf_result: Dict[str, Any]) -> List[float]:
    bus_vm = pf_result.get("bus_vm")
    if isinstance(bus_vm, list) and bus_vm:
        try:
            return [float(x) for x in bus_vm]
        except Exception:
            pass

    bus = pf_result.get("bus")
    if not bus:
        return []

    if isinstance(bus, list) and len(bus) > 0 and isinstance(bus[0], dict):
        out: List[float] = []
        for row in bus:
            for key in ("Vm_pu", "Vm", "vm", "VM", "v", "V"):
                if key in row:
                    try:
                        out.append(float(row[key]))
                        break
                    except Exception:
                        pass
        return out

    if isinstance(bus, list) and len(bus) > 0 and isinstance(bus[0], list):
        out: List[float] = []
        for row in bus:
            if len(row) > BUS_VM:
                out.append(float(row[BUS_VM]))
        return out

    return []


def vm_of_bus(pf_result: Dict[str, Any], bus_id: int) -> Optional[float]:
    """Return Vm (p.u.) of specific MATPOWER bus number."""
    bus = pf_result.get("bus")
    if not isinstance(bus, list) or not bus:
        return None

    if isinstance(bus[0], dict):
        for row in bus:
            rid = None
            for key in ("bus_i", "BUS_I", "i", "bus", "id"):
                if key in row:
                    try:
                        rid = int(row[key])
                        break
                    except Exception:
                        pass
            if rid == int(bus_id):
                for key in ("Vm_pu", "Vm", "vm", "VM", "v", "V"):
                    if key in row:
                        try:
                            return float(row[key])
                        except Exception:
                            return None
        return None

    if isinstance(bus[0], list):
        for row in bus:
            if len(row) > BUS_VM and int(row[BUS_I]) == int(bus_id):
                return float(row[BUS_VM])
        return None

    return None


# -------------------------
# Scoring (via SecurityCheck)
# -------------------------
def score_solution_via_security(
    limits: "Limits",
    base_vms: List[float],
    cand_pf: Dict[str, Any],
    cand_sc: Dict[str, Any],
    action: Dict[str, Any],
    focus_bus: Optional[int],
    base_focus_vm: Optional[float],
) -> Tuple[float, Dict[str, Any]]:
    if cand_pf.get("converged") is False:
        return -1e12, {"feasible": False, "reason": "not_converged"}

    vms = extract_vm_list(cand_pf)
    if not vms:
        return -1e12, {"feasible": False, "reason": "missing_bus_voltage"}

    ok = bool(cand_sc.get("ok", True))
    sc_score = cand_sc.get("score", None)
    if (not ok) or (sc_score is None):
        return -1e12, {"feasible": False, "reason": "security_check_failed_or_missing_score"}

    try:
        sc_score_f = float(sc_score)
    except Exception:
        return -1e12, {"feasible": False, "reason": "invalid_security_score"}

    # -------------------------
    # Base terms
    # -------------------------
    steps = abs(int(action.get("steps", 1)))
    avg_abs_dev = sum(abs(x - 1.0) for x in vms) / max(1, len(vms))

    # 主分：安全分 - 操作代价 - 全网偏离惩罚
    score = 1000.0 * sc_score_f - 5.0 * steps - 10.0 * avg_abs_dev

    base_minV = min(base_vms) if base_vms else None
    minV = min(vms)
    maxV = max(vms)

    summ = cand_sc.get("summary", {}) if isinstance(cand_sc.get("summary", {}), dict) else {}
    v_viol = summ.get("n_voltage_viol", None)
    t_viol = summ.get("n_thermal_viol", None)

    info: Dict[str, Any] = {
        "feasible": True,
        "security_score": sc_score_f,
        "security_summary": summ,
        "voltage_violations": v_viol,
        "thermal_violations": t_viol,
        "minV_after": minV,
        "maxV_after": maxV,
        "minV_before": base_minV,
        "delta_minV": (minV - base_minV) if (base_minV is not None) else None,
        "delta_viol": None,
    }

    # -------------------------
    # Focus bus improvement (distance-to-1.0 reduction)
    # improvement = |V_before-1| - |V_after-1|  (越大越好)
    # -------------------------
    if focus_bus is not None:
        cand_focus_vm = vm_of_bus(cand_pf, focus_bus)
        info["focus_bus"] = focus_bus
        info["Vm_focus_after"] = cand_focus_vm
        info["Vm_focus_before"] = base_focus_vm

        if (cand_focus_vm is not None) and (base_focus_vm is not None):
            dist_before = abs(float(base_focus_vm) - 1.0)
            dist_after  = abs(float(cand_focus_vm) - 1.0)
            focus_improve = dist_before - dist_after  # >0 表示更接近 1.0
            info["focus_dist_before"] = dist_before
            info["focus_dist_after"] = dist_after
            info["focus_improve"] = focus_improve
            info["delta_focusVm"] = float(cand_focus_vm - base_focus_vm)

            # 这项权重你可调：如果希望“目标母线优先”，就把数值加大
            W_FOCUS = 1000.0
            score += W_FOCUS * focus_improve

    return score, info

def _explain(action: Dict[str, Any], info: Dict[str, Any]) -> str:
    atype = action.get("type")
    parts = []
    if atype == "oltc":
        parts.append(f"调整 OLTC(branch_idx={action.get('branch_idx')}) tap 变化 {action.get('delta_tap'):+.4f}")
    elif atype == "shunt":
        parts.append(f"在 bus={action.get('bus')} 投切无功 {action.get('delta_mvar'):+.1f} Mvar")
    elif atype == "redispatch":
        parts.append(f"在 gen_bus={action.get('gen_bus')} 调整发电机有功 {action.get('delta_mw'):+.1f} MW")

    dm = info.get("delta_minV")
    if dm is not None:
        parts.append(f"最低电压变化 {dm:+.4f} p.u.")

    df = info.get("delta_focusVm")
    if df is not None:
        parts.append(f"目标节点电压变化 {df:+.4f} p.u.")

    dv = info.get("delta_viol")
    if dv is not None:
        try:
            parts.append(f"电压越限数量变化 {int(dv):+d}")
        except Exception:
            pass

    summ = info.get("security_summary", {})
    if isinstance(summ, dict) and int(summ.get("n_thermal_viol", 0) or 0) > 0:
        parts.append("存在线路热稳越限（由安全校核判定）")

    return "；".join(parts) if parts else "候选方案"


# -------------------------
# Text parsing
# -------------------------
def parse_voltage_obs_from_text(text: str) -> List[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return []

    # 1) JSON dict or list
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "bus" in obj and "delta" in obj:
            return [{"bus": int(obj["bus"]), "delta": float(obj["delta"])}]
        if isinstance(obj, list):
            out = []
            for it in obj:
                if isinstance(it, dict) and "bus" in it and "delta" in it:
                    out.append({"bus": int(it["bus"]), "delta": float(it["delta"])})
            if out:
                return out
    except Exception:
        pass

    # 2) Chinese pattern
    m = re.search(r"(\d+)\s*号?\s*(?:节点|母线|bus)\s*.*?(低|高)\s*.*?([0-9]*\.?[0-9]+)", t)
    if m:
        bus_id = int(m.group(1))
        sign = -1.0 if m.group(2) == "低" else +1.0
        mag = float(m.group(3))
        return [{"bus": bus_id, "delta": sign * mag}]

    # 3) fallback: bus=3, delta=-0.05
    m2 = re.search(r"bus\s*[:=]\s*(\d+).*?delta\s*[:=]\s*([+-]?[0-9]*\.?[0-9]+)", t, re.IGNORECASE)
    if m2:
        return [{"bus": int(m2.group(1)), "delta": float(m2.group(2))}]

    return []


# -------------------------
# API
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}


def _unify_observations(observation_any: Any, observations_any: Any) -> List[Dict[str, Any]]:
    """
    Normalize:
      - observation (single dict or stringified dict) -> [dict]
      - observations (list or stringified list) -> list
    """
    if observation_any is not None:
        ob = _coerce_obj(observation_any, "observation")
        if isinstance(ob, dict) and "bus" in ob and "delta" in ob:
            try:
                return [{"bus": int(ob["bus"]), "delta": float(ob["delta"])}]
            except Exception:
                return []
        return []

    obs_list_raw = _coerce_list(observations_any, "observations")
    out: List[Dict[str, Any]] = []
    for it in obs_list_raw:
        if isinstance(it, dict) and "bus" in it and "delta" in it:
            try:
                out.append({"bus": int(it["bus"]), "delta": float(it["delta"])})
            except Exception:
                pass
    return out


def _parse_req_models(req: DecisionRequest) -> Tuple[Limits, Controllables, SearchCfg, str, str, List[Dict[str, Any]], Optional[int], List[str]]:
    """
    Turn possibly-string inputs into pydantic models.
    Also fill default security_endpoint if missing.
    Return focus bus from observations (first one) and warnings.
    """
    limits_dict = _coerce_obj(req.limits, "limits")
    ctl_dict = _coerce_obj(req.controllables, "controllables")
    search_dict = _coerce_obj(req.search, "search")

    limits = Limits(**limits_dict)
    controllables = Controllables(**ctl_dict)
    search = SearchCfg(**search_dict)

    pf_url = str(req.pf_endpoint)
    sc_url = str(req.security_endpoint) if req.security_endpoint is not None else DEFAULT_SECURITY_ENDPOINT

    observations = _unify_observations(req.observation, req.observations)
    focus_bus = int(observations[0]["bus"]) if observations else None

    warnings: List[str] = []
    if observations:
        dv0 = float(observations[0]["delta"])
        if req.issue == "voltage_low" and dv0 > 0:
            warnings.append("issue=voltage_low but observation delta>0 (possible conflict).")
        if req.issue == "voltage_high" and dv0 < 0:
            warnings.append("issue=voltage_high but observation delta<0 (possible conflict).")

    return limits, controllables, search, pf_url, sc_url, observations, focus_bus, warnings


@app.post("/decision/voltage_support")
def voltage_support(req: DecisionRequest):
    try:
        limits, controllables, search, pf_url, sc_url, observations, focus_bus, warnings = _parse_req_models(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        base_case = load_ieee_case_from_id(req.case_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # inject observations into case (create scenario)
    if observations:
        base_case = apply_voltage_observations_to_case(
            base_case,
            observations=observations,
            q_sensitivity_mvar_per_pu=search.q_sensitivity_mvar_per_pu,
            p_sensitivity_mw_per_pu=search.p_sensitivity_mw_per_pu,
        )

    # baseline PF
    try:
        base_pf = call_run_pf(pf_url, req.method, base_case, timeout_sec=search.pf_timeout_sec)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"baseline run_pf failed: {e}")

    base_vms = extract_vm_list(base_pf)
    base_focus_vm = vm_of_bus(base_pf, focus_bus) if focus_bus is not None else None

    # baseline SecurityCheck
    try:
        base_sc = call_security_check(sc_url, base_pf, limits, timeout_sec=search.pf_timeout_sec)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"baseline security_check failed: {e}")

    base_summ = base_sc.get("summary", {}) if isinstance(base_sc.get("summary", {}), dict) else {}
    base_v_viol = base_summ.get("n_voltage_viol", None)

    # generate candidates
    cands = generate_candidates(req.issue, controllables, search.max_candidates)
    if not cands:
        return {
            "status": "ok",
            "case_id": req.case_id,
            "issue": req.issue,
            "observation": (observations[0] if observations else None),
            "observations": observations,
            "warnings": warnings,
            "message": "no candidates generated (check controllables.*.enabled and indices)",
            "baseline": {
                "pf": {"converged": base_pf.get("converged", None)},
                "v_summary": {
                    "minV": (min(base_vms) if base_vms else None),
                    "maxV": (max(base_vms) if base_vms else None),
                    "focus_bus": focus_bus,
                    "Vm_focus": base_focus_vm,
                },
                "security": base_sc,
            },
            "topk": [],
        }

    results: List[Dict[str, Any]] = []

    def _eval_one(action: Dict[str, Any]) -> Dict[str, Any]:
        mod_case = apply_action(base_case, action, controllables)
        try:
            pf = call_run_pf(pf_url, req.method, mod_case, timeout_sec=search.pf_timeout_sec)
            sc = call_security_check(sc_url, pf, limits, timeout_sec=search.pf_timeout_sec)

            sc_score, info = score_solution_via_security(
                limits=limits,
                base_vms=base_vms,
                cand_pf=pf,
                cand_sc=sc,
                action=action,
                focus_bus=focus_bus,
                base_focus_vm=base_focus_vm,
            )

            cand_summ = sc.get("summary", {}) if isinstance(sc.get("summary", {}), dict) else {}
            cand_v_viol = cand_summ.get("n_voltage_viol", None)
            if base_v_viol is not None and cand_v_viol is not None:
                try:
                    info["delta_viol"] = int(base_v_viol) - int(cand_v_viol)
                except Exception:
                    pass

            vms = extract_vm_list(pf)
            cand_focus_vm = vm_of_bus(pf, focus_bus) if focus_bus is not None else None

            return {
                "action": action,
                "score": sc_score,
                "pf": {"converged": pf.get("converged", None)},
                "security": sc,
                "improvement": info,
                "v_detail": {
                    "focus_bus": focus_bus,
                    "Vm_focus": cand_focus_vm,
                    "minV": (min(vms) if vms else None),
                    "maxV": (max(vms) if vms else None),
                },
                "explain": _explain(action, info),
                "error": None,
            }
        except Exception as e:
            info = {"feasible": False, "reason": f"eval_failed: {e}"}
            return {
                "action": action,
                "score": -1e12,
                "pf": {"converged": None},
                "security": None,
                "improvement": info,
                "v_detail": {
                    "focus_bus": focus_bus,
                    "Vm_focus": None,
                    "minV": None,
                    "maxV": None,
                },
                "explain": _explain(action, info),
                "error": str(e),
            }

    max_workers = max(1, int(search.max_parallel_pf))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_eval_one, a) for a in cands]
        for f in as_completed(futs):
            results.append(f.result())

    results.sort(key=lambda x: (x["improvement"].get("feasible", False), x["score"]), reverse=True)
    topk_n = max(1, int(search.topk))
    topk = [{"rank": i + 1, **item} for i, item in enumerate(results[:topk_n])]

    return {
        "status": "ok",
        "case_id": req.case_id,
        "issue": req.issue,
        "observation": (observations[0] if observations else None),
        "observations": observations,
        "warnings": warnings,
        "baseline": {
            "pf": {"converged": base_pf.get("converged", None)},
            "v_summary": {
                "minV": (min(base_vms) if base_vms else None),
                "maxV": (max(base_vms) if base_vms else None),
                "focus_bus": focus_bus,
                "Vm_focus": base_focus_vm,
            },
            "security": base_sc,
        },
        "topk": topk,
        "notes": "Ranking: feasible(converged+security ok) > higher security_score > lower action cost",
    }


@app.post("/decision/voltage_support_debug")
def voltage_support_debug(req: DecisionDebugRequest):
    try:
        limits_dict = _coerce_obj(req.limits, "limits")
        ctl_dict = _coerce_obj(req.controllables, "controllables")
        search_dict = _coerce_obj(req.search, "search")

        limits = Limits(**limits_dict)
        controllables = Controllables(**ctl_dict)
        search = SearchCfg(**search_dict)

        pf_url = str(req.pf_endpoint)
        sc_url = str(req.security_endpoint) if req.security_endpoint is not None else DEFAULT_SECURITY_ENDPOINT

        observations = _unify_observations(req.observation, req.observations)
        focus_bus = int(observations[0]["bus"]) if observations else None

        warnings: List[str] = []
        if observations:
            dv0 = float(observations[0]["delta"])
            if req.issue == "voltage_low" and dv0 > 0:
                warnings.append("issue=voltage_low but observation delta>0 (possible conflict).")
            if req.issue == "voltage_high" and dv0 < 0:
                warnings.append("issue=voltage_high but observation delta<0 (possible conflict).")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        base_case = normalize_matpower_case(req.case)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid case: {e}")

    if observations:
        base_case = apply_voltage_observations_to_case(
            base_case,
            observations=observations,
            q_sensitivity_mvar_per_pu=search.q_sensitivity_mvar_per_pu,
            p_sensitivity_mw_per_pu=search.p_sensitivity_mw_per_pu,
        )

    try:
        base_pf = call_run_pf(pf_url, req.method, base_case, timeout_sec=search.pf_timeout_sec)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"baseline run_pf failed: {e}")

    base_vms = extract_vm_list(base_pf)
    base_focus_vm = vm_of_bus(base_pf, focus_bus) if focus_bus is not None else None

    try:
        base_sc = call_security_check(sc_url, base_pf, limits, timeout_sec=search.pf_timeout_sec)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"baseline security_check failed: {e}")

    base_summ = base_sc.get("summary", {}) if isinstance(base_sc.get("summary", {}), dict) else {}
    base_v_viol = base_summ.get("n_voltage_viol", None)

    cands = generate_candidates(req.issue, controllables, search.max_candidates)
    if not cands:
        return {
            "status": "ok",
            "message": "no candidates",
            "observation": (observations[0] if observations else None),
            "observations": observations,
            "warnings": warnings,
            "baseline": {
                "pf": {"converged": base_pf.get("converged", None)},
                "v_summary": {
                    "minV": (min(base_vms) if base_vms else None),
                    "maxV": (max(base_vms) if base_vms else None),
                    "focus_bus": focus_bus,
                    "Vm_focus": base_focus_vm,
                },
                "security": base_sc,
            },
            "topk": [],
        }

    results: List[Dict[str, Any]] = []
    for action in cands:
        mod_case = apply_action(base_case, action, controllables)
        try:
            pf = call_run_pf(pf_url, req.method, mod_case, timeout_sec=search.pf_timeout_sec)
            sc = call_security_check(sc_url, pf, limits, timeout_sec=search.pf_timeout_sec)

            sc_score, info = score_solution_via_security(
                limits=limits,
                base_vms=base_vms,
                cand_pf=pf,
                cand_sc=sc,
                action=action,
                focus_bus=focus_bus,
                base_focus_vm=base_focus_vm,
            )

            cand_summ = sc.get("summary", {}) if isinstance(sc.get("summary", {}), dict) else {}
            cand_v_viol = cand_summ.get("n_voltage_viol", None)
            if base_v_viol is not None and cand_v_viol is not None:
                try:
                    info["delta_viol"] = int(base_v_viol) - int(cand_v_viol)
                except Exception:
                    pass

            vms = extract_vm_list(pf)
            cand_focus_vm = vm_of_bus(pf, focus_bus) if focus_bus is not None else None

            results.append({
                "action": action,
                "score": sc_score,
                "pf": {"converged": pf.get("converged", None)},
                "security": sc,
                "improvement": info,
                "v_detail": {
                    "focus_bus": focus_bus,
                    "Vm_focus": cand_focus_vm,
                    "minV": (min(vms) if vms else None),
                    "maxV": (max(vms) if vms else None),
                },
                "explain": _explain(action, info),
                "error": None,
            })
        except Exception as e:
            info = {"feasible": False, "reason": str(e)}
            results.append({
                "action": action,
                "score": -1e12,
                "pf": {"converged": None},
                "security": None,
                "improvement": info,
                "v_detail": {
                    "focus_bus": focus_bus,
                    "Vm_focus": None,
                    "minV": None,
                    "maxV": None,
                },
                "explain": _explain(action, info),
                "error": str(e),
            })

    results.sort(key=lambda x: (x["improvement"].get("feasible", False), x["score"]), reverse=True)
    topk = [{"rank": i + 1, **item} for i, item in enumerate(results[: search.topk])]

    return {
        "status": "ok",
        "observation": (observations[0] if observations else None),
        "observations": observations,
        "warnings": warnings,
        "baseline": {
            "pf": {"converged": base_pf.get("converged", None)},
            "v_summary": {
                "minV": (min(base_vms) if base_vms else None),
                "maxV": (max(base_vms) if base_vms else None),
                "focus_bus": focus_bus,
                "Vm_focus": base_focus_vm,
            },
            "security": base_sc,
        },
        "topk": topk,
    }


@app.post("/decision/voltage_support_from_text")
def voltage_support_from_text(req: TextObsRequest):
    obs = parse_voltage_obs_from_text(req.text)
    if not obs:
        raise HTTPException(status_code=400, detail="cannot parse voltage observation from text")

    issue = req.issue
    if issue is None:
        issue = "voltage_low" if float(obs[0]["delta"]) < 0 else "voltage_high"

    dr = DecisionRequest(
        pf_endpoint=req.pf_endpoint,
        security_endpoint=req.security_endpoint,
        method=req.method,
        case_id=req.case_id,
        issue=issue,
        observation=obs[0],   # ✅ 用单对象入口（更适配 Dify）
        limits=req.limits,
        controllables=req.controllables,
        search=req.search,
    )
    return voltage_support(dr)


# -------------------------
# Keep /openapi_dify.json
# -------------------------
from pathlib import Path
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent


@app.get("/openapi_dify.json", include_in_schema=False)
def openapi_dify():
    return FileResponse(str(BASE_DIR / "openapi_dify.json"), media_type="application/json")
