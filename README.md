# VoltageDecisionSupportTool

`VoltageDecisionSupportTool` 是一个基于 FastAPI 的电压决策支持工具，用于电力系统中的电压问题诊断和解决方案生成。它集成了电力潮流计算和安全检查功能，能够根据输入的电力系统状态（如电压偏低或偏高的情况）自动生成可能的调节方案（如调节变压器抽头、无功补偿、重新分配发电机功率等）。

该工具主要应用于配电网运行中电压控制和无功管理问题，结合了现代电力系统仿真技术与智能决策支持方法。

## 主要功能

1. **电压问题诊断**：检查电力系统中每个母线的电压是否超出规定的上限和下限，判断是否存在电压偏高或偏低问题。
2. **热稳定性检查**：评估电力系统支路的功率流是否超过热稳限制（如支路的额定容量）。
3. **决策支持**：根据电力系统状态（例如电压偏低或偏高），生成相应的调节方案，并优化调整策略。可以调整变压器抽头（OLTC）、无功补偿（Shunt）或重新分配发电机功率（Redispatch）。
4. **基于文本的输入**：支持通过自然语言描述电压偏差（例如“3号节点电压低了0.05 p.u.”），自动解析并生成相应的决策请求。
5. **性能优化**：支持并行计算，能够处理多个候选方案，并根据电力潮流和安全检查结果进行评分排序，提供最佳解决方案。

## 安装

首先，克隆该仓库并创建虚拟环境：

```bash
# 克隆仓库
git clone <your-repository-url>

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # 对于Windows用户：venv\\Scripts\\activate

# 安装依赖
pip install -r requirements.txt
运行应用
# 启动 FastAPI 应用
uvicorn decision_service.app:app --reload
访问地址：<http://127.0.0.1:8000>

```

## API 接口

### 健康检查 (/health)

响应：

```json
{
  "ok": true
}
```

GET /health

### 电压支持决策 (`/decision/voltage_support`)

- **POST** `/decision/voltage_support`

### 请求体

```json
{
  "pf_endpoint": "http://<your-power-flow-tool-url>/run_pf",
  "security_endpoint": "http://<your-security-check-url>/security_check",
  "method": "ac",
  "case_id": "case14",
  "issue": "voltage_low",
  "observation": {"bus": 3, "delta": -0.05},
  "limits": {
    "vmin": 0.95,
    "vmax": 1.05,
    "thermal_limit_field": "rateA"
  },
  "controllables": {
    "redispatch": {
      "enabled": true,
      "gen_buses": [1, 2],
      "step_mw": 10.0,
      "max_steps": 3
    },
    "shunt": {
      "enabled": true,
      "buses": [1, 2],
      "step_mvar": 5.0,
      "max_steps": 3
    },
    "oltc": {
      "enabled": true,
      "branch_idx": [0, 1],
      "step_tap": 0.0125,
      "max_steps": 2
    }
  },
  "search": {
    "max_candidates": 30,
    "topk": 5,
    "max_parallel_pf": 8,
    "pf_timeout_sec": 20
  }
}
```

### 响应

```json
{
  "status": "ok",
  "case_id": "case14",
  "issue": "voltage_low",
  "observations": [{"bus": 3, "delta": -0.05}],
  "warnings": ["issue=voltage_low but observation delta>0 (possible conflict)"],
  "baseline": {
    "pf": {"converged": true},
    "v_summary": {"minV": 0.95, "maxV": 1.05, "focus_bus": 3, "Vm_focus": 0.96},
    "security": {"score": 0.95, "summary": {"n_voltage_viol": 0, "n_thermal_viol": 0}}
  },
  "topk": [
    {
      "rank": 1,
      "score": 0.98,
      "action": {"type": "oltc", "branch_idx": 0, "delta_tap": 0.0125, "steps": 1},
      "explain": "调整 OLTC(branch_idx=0) tap 变化 +0.0125",
      "v_detail": {"focus_bus": 3, "Vm_focus": 0.96, "minV": 0.95, "maxV": 1.05}
    }
  ]
}
```

### 通过文本进行电压支持决策 (`/decision/voltage_support_from_text`)

- **POST** `/decision/voltage_support_from_text`

### 请求体

```json
{
  "text": "3号节点电压低了0.05 p.u.",
  "pf_endpoint": "http://<your-power-flow-tool-url>/run_pf",
  "security_endpoint": "http://<your-security-check-url>/security_check",
  "method": "ac",
  "case_id": "case14"
}
```

### 响应

响应格式与 `/decision/voltage_support` 相同。

## 依赖

- **FastAPI**: Web框架，用于处理HTTP请求。
- **Pydantic**: 用于数据验证和模型定义。
- **Requests**: HTTP 客户端库，用于与外部API通信。
- **pypower**: 用于加载 IEEE 电力系统案例（如 `case14` 和 `case30`）。

## 贡献

欢迎提交 PR，报告问题或提供反馈。我们鼓励开源贡献，任何有助于改进本工具的想法都非常欢迎。
