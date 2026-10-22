# 最终验收报告

日期：2026-10-22。目标：在保留现有七阶段架构的前提下，审计、补齐并实测科研论文多轮检索问答 Agent，交付可复现、可追踪、可比较的评测闭环。本报告按「汇报格式」逐项说明。

## 1. 实际修改的文件及原因

### 新增
| 文件 | 原因 |
|---|---|
| `docs/implementation-audit.md` | 阶段 A 审计报告：七阶段逐项核对、缺陷与修改建议 |
| `src/paper_agent/judge.py` | LLM-as-Judge（6 维结构化评分 + 调用记录，隐藏实验组标签） |
| `src/paper_agent/eval_tasks.py` | 意图/改写/边界/记忆评测指标与任务运行器 |
| `src/paper_agent/eval_cli.py` | 统一 `eval` 子命令族 |
| `configs/evaluation.json` | 统一评测配置（数据集、模型、Top-K、Reflect/记忆开关、seed、Judge） |
| `data/eval/`（6 个 jsonl + README + manifest） | 阶段 D 自建数据集，明确标签来源与审核状态 |
| `scripts/api_smoke.py`、`scripts/api_diag_text.py` | 真实 API smoke test 与 400 诊断 |
| `scripts/prepare_eval.py` | 生成 `data/eval/`（无模型调用） |
| `scripts/verify_c_fixes.py` | 阶段 C 修复的独立验证 |
| `reports/LIVE_EVAL_RESULTS.md` | 真实 API 评测结果记录 |

### 修改
| 文件 | 变更 |
|---|---|
| `src/paper_agent/pipeline.py` | Guardrails 增控制字符校验；引用校验失败改为受控拒答（`citation_validation_failed`）；语义压缩失败回退到确定性裁剪（`compaction_fallback`） |
| `src/paper_agent/cli.py` | `eval` 改为子命令族，保留 ask/chat/search/shell 交互入口 |
| `tests/test_core.py` | 新增 3 个回归测试（控制字符/引用拒答/压缩回退） |
| `docs/api.md`、`docs/generation.md`、`docs/memory.md`、`docs/evaluation.md`、`docs/README.md`、`README.md` | 与真实源码和实测结果对齐 |

## 2. 核心架构是否符合参考截图

七阶段 `Guardrails → NLU → Retrieve → Rerank → Reflect → Generate → Memory` 与参考截图一致，保留 `PaperAgentState`、`PaperPipeline.run()`、`TurnTrace`、检索/记忆模块整体架构，未做结构性重构。二次检索上限、引用 ID 校验、会话记忆与归档均保留。新增的是统一的评测层（eval 子命令 + Judge + data/eval），不改变主流程。

## 3. 各模块实现与验证状态

| 模块 | 实现 | 验证方式 |
|---|---|---|
| Guardrails | 空/超长/控制字符校验，入 trace | 新增测试 + 独立脚本 |
| NLU | 五分类 + 指代 + 论文范围 + 澄清 | 真实 API（25 条意图 + 4 轮会话） |
| Retrieve/Rerank | BM25/FAISS/RRF/Cross-Encoder | 真实 GPU 四组基线 |
| Reflect | answer/retrieve_again/clarify/refuse + 二次检索上限 | 离线测试 + 真实会话（t3 触发二次检索） |
| Generate | claims + citations + 结构校验 | 真实 API；引用失败改受控拒答 |
| Memory | 会话隔离/事实/实体/归档/两种压缩 | 离线测试 + 压缩回退验证 |
| 可观测 | TurnTrace + api_calls + usage | 真实 trace 已落盘 |
| 评测 | eval 子命令 + Judge + data/eval | 真实运行（intent/e2e） |

## 4. OpenCode 真实 API 调用情况

- 配置：`API_BASE_URL=https://opencode.ai/zen/go/v1`，`MODEL_ID=deepseek-v4.1-flash`（`.env` 已配置，未改凭证）。
- Smoke test **通过**：文本生成 + JSON 结构化输出，`model_returned` 一致，usage 可用（`reports/api-smoke/latest.json`）。
- 关键发现：①端点每次请求要求 `x-opencode-session` 头（缺失→400）；②`deepseek-v4.1-flash` 是推理模型，`reasoning_tokens` 计入 `max_tokens`，2048 预算对复杂比较题/长答案不足（见 §5）。

## 5. 评测指标、基线与置信区间

### 检索基线（既有 GPU 结果，未变更，Recall@5，已知论文范围，段落覆盖代理）
| 配置 | Recall@5 | Hit@5 |
|---|---:|---:|
| BM25 | 66.26% | 68.00% |
| Dense | 73.94% | 78.00% |
| Hybrid+RRF | 74.24% | 77.00% |
| Hybrid+RRF+Rerank | 77.61% | 81.00% |

hybrid−dense +0.30pp，rerank−hybrid +3.38pp；按论文聚类的配对 bootstrap 95% 区间均跨零（`reports/qasper-suite/paired_comparison.json`），不声称显著。

### 意图分类（25 条平衡子集，draft 标签，`reports/runs/intent-25-*/metrics.json`）
Accuracy 84.0%（21/25），Macro-F1 0.837；混淆集中在 fact↔search 与 fact→out_of_scope。**这是与草稿标签的一致率，不是人工 gold 准确率。**

### 端到端 4 轮（`reports/runs/e2e-d01-*/`）
3/4 完成（t1 answer、t2 answer 且指代正确、t4 越界正确拒答），t3 跨论文比较 generate 截断（`finish_reason=length`，`reasoning_tokens=2048` 耗尽预算）。Judge 2/3 成功（t1、t4 全维度 5 分，t4 `appropriate_abstention` 正确）。

**结论**：对推理模型应提高 `API_MAX_TOKENS`（≥4096）或设 `API_THINKING=disabled`；当前 2048 对简单问题足够，对长/复杂答案不足。

## 6. 记忆压缩的质量与 token 成本对比

**未正式评测**。原因：需要长会话压力集（30×4 默认不触发压缩），且需人工核对事实保留与 token 成本。本次仅验证了语义压缩失败回退到确定性裁剪的机制（`verify_c_fixes.py`）。截图中的 80.3% 一致性 / 34% token 降幅保持「待复测」，未填入本项目结果。

## 7. 未完成任务与原因

1. **人工复核标签**：意图/边界/改写/记忆标签为 `draft_unreviewed`，需独立审阅后才能作为 gold（任务要求，无法由 agent 替代人工）。
2. **记忆压缩质量+token 对比**：需长会话压力集与人工事实核对。
3. **Reflect 消融、完整 70-turn 端到端、多轮批量**：未运行（API 配额与时间；且草稿标签未复核，结果不具 gold 意义）。
4. **git commit 记录**：仓库无 `.git`，无法提供 commit hash。
5. **pytest 在本沙箱无法运行**：`tmp_path` fixture 因工作区继承的 ACL（`Low Mandatory Level` + DENY）无法列出临时目录；代码本身在先前环境 29 测试通过（`reports/implementation-tests.xml`），本次新增 3 测试与 3 处修复通过独立脚本验证。

## 8. 从干净环境复现

```powershell
python -m pip install -e '.[models,api,dev]'
python -m pytest -q                       # 29 + 3 新增测试
python scripts/prepare_qasper.py --papers 100 --queries 100
python scripts/prepare_eval.py            # 生成 data/eval/
python scripts/run_retrieval_suite.py --device cuda:0   # 四组检索基线
python -m paper_agent.cli eval smoke --env-file .env
python -m paper_agent.cli eval intent data/eval/intent.jsonl --output runs/intent --env-file .env
python -m paper_agent.cli eval e2e data/eval/d01-4turns.jsonl --index indexes/bge --mode rerank --judge --env-file .env --output runs/e2e
```

所有运行产物保留在 `reports/runs/`、`reports/api-smoke/`，历史报告不覆盖。
