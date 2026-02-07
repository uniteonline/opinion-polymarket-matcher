# Opinion-Polymarket Matcher (Discovery Only)

Opinion 与 Polymarket 的市场匹配/发现服务（仅保留匹配逻辑，不包含交易/对冲引擎）。
A discovery and matching service for Opinion ↔ Polymarket markets (matching only, no trading/hedging logic).

## 功能 / Features
- 从 Opinion 与 Polymarket 拉取市场快照并进行匹配
- 标题门控 + 子标题门控 + 时间窗 + 标签/队伍特征打分
- 生成匹配结果、低置信度列表与 gate 过滤记录
- 结果写入 SQLite 便于审查与下游消费

## 实现方式 / How It Works
1. 拉取 Opinion 与 Polymarket 市场
2. 规则过滤（活跃状态、二元市场、时间窗）
3. 计算相似度与综合评分，挑选最佳匹配
4. 写入 `discovery.db` 并记录低置信度/未通过 gate 的候选

## 技术栈 / Tech Stack
- Python 3.10+
- asyncio + aiohttp
- aiosqlite (SQLite)
- pyyaml

## 安装 / Installation
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 配置 / Configuration
复制模板并填写：
```bash
cp discovery/config.example.yaml discovery/config.yaml
```

设置 Opinion API Key（推荐环境变量）：
```bash
export OPINION_OPENAPI_KEY="YOUR_OPINION_API_KEY"
```

> `discovery/config.yaml` 中的 `opinion.api_key` 为空时，会自动读取 `OPINION_OPENAPI_KEY`。

## 使用 / Usage
一次性运行：
```bash
python -m discovery.run --config discovery/config.yaml --once
```

定时循环：
```bash
python -m discovery.run --config discovery/config.yaml
```

导出低置信度审查列表：
```bash
python -m discovery.review_cli --config discovery/config.yaml
```

## 输出 / Outputs
- `data/discovery.db`
  - `market_pairs`: 匹配结果
  - `low_confidence_pairs`: 低置信度候选
  - `gate_filtered_pairs`: 未通过 gate 的候选

## Notes
- 该仓库仅保留市场匹配与发现逻辑，不包含交易/对冲模块。
- 请勿提交真实 API Key。
