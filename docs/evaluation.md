当前修订验证见 [revision/FINAL](../reports/revision/FINAL.md)，固定低成本实验使用 scripts/revision_experiments.py；规则标签、替身、真实服务与模型质量分别报告。

2026-10-28 跟进的两项修复、受限与正常环境测试及新的三组模型配对见 [followup](../reports/revision/FOLLOWUP.md)。原 `model-pair/summary.json` 的 `budget.actual_requests=6` 实际表示预算上限，逐组实际2+3=5；原始文件未改。新的实验使用 `max_actual_requests` 和 `actual_requests_total` 分列。

最终多轮范围修复、当前源码真实三存储复验及六例多轮对照见 [closure](../reports/revision/CLOSURE.md)。两条实际前置对话和各条件的模型调用分别计入；作者构造的目标标签、官方证据段落和 unknown 语义质量分开报告。

Langfuse 可选投影和本轮验证状态见 [Langfuse 阶段报告](../reports/revision/LANGFUSE.md)。`eval e2e` 在逐例预测中保存整个输入集的哈希、无凭据配置及其哈希，在规则 Score 元数据中关联这些哈希、评分来源和伪匿名 case/run 标识。每组使用全新的输出和记忆目录，避免同 turn 幂等重放引用其他运行。`unknown` 本地保留，不发送数值；如果远端先前已有已知分数而本地改为 unknown，旧分数可能残留，不能以远端视图取代本地结果。同模型 Judge 明确标记，不是人工准确率。候选覆盖不代表证据语义支持。

旧数据集中的 `facts`、`fact_source_papers` 检查依赖已退役的表示；新运行将这些检查记为 unknown，检查点分数为不将 unknown 视为成功的下界，不将空值冒充观测结果。旧冻结报告仍保留原始口径。

# 当前评测与复现

修订前冻结验收结论：[FINAL_ACCEPTANCE.md](../reports/FINAL_ACCEPTANCE.md)。主配置为 `configs/final-acceptance.json`：真实 BGE-M3/FAISS、BM25/RRF、Cross-Encoder 与 three_layer 记忆。默认 CLI 的 BM25/legacy 不代表该配置。

## 数据与指标

| 项目 | 数据与计分 |
|---|---|
| Intent | v0.6，100 条来源规则标签；不冒充人工金标 |
| Rewrite | v0.6，90 条；60 条论文指代主视图，指代/含义约束/独立检索分别判断 |
| Retrieval | 100 条 QASPER 已知论文问题；四组统一 chunks、候选预算与段落覆盖 Recall@5 |
| Boundary | 55 条；仅问题及完整历史完全一致时复用本轮 E2E；决策与拒答 P/R、误拒答分列 |
| E2E | 30×4=120 轮；70-turn 为执行完整历史后的固定计分视图 |
| Citation | 最终输出 ID/来源合法率与同模型语义支持率分开，unknown 留在分母 |
| Memory | 3 场景×9 轮，full/bounded；事实值、更新、版本、冲突、事件召回契约 |
| Reflect | 固定12轮×3设置，另6例×3诊断；自然效果与强制分支测试分开 |
| Token | 实际 usage 包含生成、反思、摘要与重试；Judge 另计，缺失 usage 不按零 |

正式结果目录 `reports/final-acceptance/live-v2`。Intent/Rewrite 复用本次早期已完整完成的 NLU-only 结果；检索复用2026-10-26独立重跑，校验记录在 `retrieval-reused/`。失败、中断和重跑保留，不覆盖成一次成功实验。

## 成功判定与限制

`eval_protocol.assess` 区分 execution、behavior、task、pipeline。未提供独立答案质量时为 unknown，不凭引用存在推断正确。模型辅助 task_correct 单列，不填充人工 reviewed 字段。

检索 Recall 以 gold 段落覆盖宏平均、多标注最佳替代路径计分；命中长段落某个窗口仍只是覆盖代理。全库检索与 known-paper 不是同一任务。语义支持 Judge 与生成模型相同，尚无用户人工抽查。

本次 bounded token 增加1.20%，Reflect 未证明稳定增益。Memory 契约通过率不等于任意自然对话准确率。完整失败、unknown 和30例复核包随报告保存。

## 复现顺序

已有资产直接复用。新环境见 [安装文档](setup.md)。Redis 按本机实际地址启动，本机隔离服务端口为16379；`.env` 不覆盖。先本地检查，再付费实验：

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'
& $py -m pytest -q
& $py scripts/check_artifacts.py
& $py scripts/check_memory_integration.py --redis-url redis://localhost:16379/0 --output reports/reproduced-integration
# 以下使用 API；选择新的输出目录，core 完成后再运行 judge。
& $py scripts/run_final_acceptance.py --output reports/reproduced-live --config configs/final-acceptance.json
& $py scripts/run_final_acceptance.py --output reports/reproduced-live --config configs/final-acceptance.json --phase judge
& $py scripts/reflect_diagnostic.py --output reports/reproduced-reflect
```

每个请求先落盘占用预算，失败、重试、Judge 均计入。core/judge 不得同时写同一 output。冻结配置/源码/数据不一致会拒绝恢复；半途会话保留，另建输出目录，不拼接冒充完整运行。

`run_final_acceptance.py` 对新输出重新运行NLU，并受配置的请求上限限制；原实验中复用NLU带来的成本不能当作每次新运行保证。先看 estimated_requests 与剩余配额，预算不足明确保留失败，不自动扩额。

汇总工具 `summarize_final_acceptance.py`、`write_final_report.py` 是当前交付快照的生成器，固定读取 `reports/final-acceptance/`；不要用它们覆盖一轮新实验冒充原结果。新运行应单独记录配置和报告。

## 历史实验

Phase A 的 BM25/legacy 结果、旧 acceptance、早期检索和错误修复日志已归入 [reports/history/development](../reports/history/development/)。路径迁移表见 [relocations.json](../reports/history/relocations.json)。历史原始记录中的命令与路径保留当时值，相关开发脚本读取路径已调整。旧 v0.3 Reflect 子集仍用于当前对照，不能删除。

`eval compare` 支持按会话/论文配对的簇 bootstrap；少量簇的区间不能作为总体显著性证据。增加统计框架不是当前工程验收条件。
