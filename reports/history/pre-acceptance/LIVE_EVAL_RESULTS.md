# 真实 API 评测记录（Live Evaluation）

日期：2026-10-22。本文件记录用 `.env` 中配置的 OpenCode API（`https://opencode.ai/zen/go/v1`，模型 `deepseek-v4.1-flash`）完成的真实调用与评测结果。所有数字来自 `reports/api-smoke/`、`reports/runs/` 下的实际产物，不包含 API key。

## 1. API Smoke Test

结果文件：`reports/api-smoke/latest.json`。

| 检查 | 结果 |
|---|---|
| 简单文本生成 | ✅ `content="pong"`，`finish_reason=stop`，`model_returned=deepseek-v4.1-flash` |
| JSON 结构化输出（NLU） | ✅ 合法 intent/standalone_query，`validation=ok` |
| usage 可用性 | ✅ 真实返回 `prompt_tokens`/`completion_tokens`/`prompt_tokens_details.cached_tokens` |
| 模型标识 | ✅ 返回模型与请求模型一致 |

关键发现：

- OpenCode `/zen/go/v1` 端点在**每次请求都要求 `x-opencode-session` 头**；缺失返回 `400 MissingSessionID`。`CompatibleReasoner` 已固定发送该头。
- `deepseek-v4.1-flash` 是推理模型：响应含 `reasoning_content` 与 `content`，`completion_tokens` 包含 `reasoning_tokens`。reasoner 只读取 `content` 作为结构化结果。

## 2. 意图分类（25 条平衡子集，草稿标签）

结果目录：`reports/runs/intent-25-20261022-175022/`。

| 指标 | 值 |
|---|---|
| Accuracy | 84.0%（21/25） |
| Macro-F1 | 0.837 |
| 每类 F1 | compare 1.0、explain 1.0、out_of_scope 0.83、search 0.75、fact 0.60 |

混淆集中在 fact↔search 与 fact→out_of_scope：这源于 QASPER 派生的「事实问题」与「检索问题」在语义上高度重叠。**注意：这是与 `data/eval/intent.jsonl` 中 `draft_unreviewed` 临时标签的一致率，不是人工 gold 准确率。**

## 3. 端到端 4 轮会话（session d01，rerank + Judge）

结果目录：`reports/runs/e2e-d01-20261022-175339/`（`metrics.json`、`predictions.jsonl`、`judge/judgments.json`、`traces/d01/`）。

| turn | 预期行为 | 实际结果 |
|---|---|---|
| t1 | 单篇论文事实检索 | ✅ answer（英文，3 claims，引用校验通过） |
| t2 | 指代追问（“刚才那篇论文”） | ✅ answer（中文，7 claims，指代正确解析到 paper 1809.02208） |
| t3 | 跨论文比较 | ❌ generate 截断（`finish_reason=length`） |
| t4 | 越界（私人银行账户） | ✅ refuse（`out_of_scope` 正确处理） |

Pipeline 完成率 3/4；Judge 判定 2/3 成功（t1、t4 均全维度 5 分，t4 的 `appropriate_abstention.applicable=true` 正确识别了“正确拒答”）。

### 3.1 重要发现：推理模型的 token 预算

t3 与 t2 的 Judge 失败根因相同：`deepseek-v4.1-flash` 把 `reasoning_tokens` 计入 `max_tokens`。当问题复杂（跨论文比较）或答案较长（7 claims）时，模型把全部 2048 输出预算花在推理上，`content` 为空/截断，`finish_reason=length`，provider 按 fail-closed 抛错（这是正确行为，不是代码 bug）。

- t3 generate：`completion_tokens=2048`，其中 `reasoning_tokens=2048`，实际 JSON 答案无预算。
- t2 judge：`completion_tokens=2048`，其中 `reasoning_tokens=2048`。

**结论与建议**：对推理模型应调高 `API_MAX_TOKENS`（建议 ≥4096，复杂比较题考虑 8192），或关闭 `API_THINKING`（`disabled`）以省去推理 token。当前 `.env` 的 `API_MAX_TOKENS=2048` 对简单事实/检索问题足够，对长答案和比较题不足。该限制已记录，不作为“系统失败”，而是可配置的预算约束。

## 4. 检索四组基线（既有离线 GPU 结果，未变更）

见 [qasper-suite/RESULTS.md](../../qasper-suite/RESULTS.md)：BM25 66.26% / dense 73.94% / hybrid 74.24% / rerank 77.61%（Recall@5，已知论文范围，段落覆盖代理指标，配对 bootstrap 区间跨零）。

## 5. 尚未完成 / 已知限制

- 意图/边界/改写/记忆标签均为 `draft_unreviewed`，未人工复核；不能作为效果 gold。
- Judge 是同模型辅助判定，非人工验证；需抽人工复核并报告一致率。
- 记忆压缩的 token 降幅与一致性、Reflect 的消融效果尚未正式评测（需长会话压力集）。
- 无 git 仓库，无法记录 commit；环境与依赖版本见 `reports/environment-freeze.txt`。
