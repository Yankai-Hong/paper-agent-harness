# 当前验收证据

总览见 [最终报告](../FINAL_ACCEPTANCE.md)。

| 文件/目录 | 用途 |
|---|---|
| `summary.json` | 所有指标、分母、Trace统计和成本 |
| `live-v2/freeze.json` | 当前运行配置、源码/数据哈希 |
| `live-v2/e2e/` | 完整120轮结果、Trace和评审，70轮视图由此取样 |
| `live-v2/intent/`、`rewrite/`、`boundary/` | 组件结果与明确复用来源 |
| `live-v2/memory-*`、`reflect-*` | 长记忆与反思对照 |
| `retrieval-reused/` | 四组检索结果及复用依赖校验 |
| `integration-v2/` | 固定回答器真实存储/重排/恢复集成 |
| `reflect-diagnostic-v2/` | 六例自然诊断与强制分支的独立记录 |
| `claims.jsonl`、`memory-checks.jsonl` | 引用和具体记忆契约证据 |
| `failures.jsonl`、`review-package.jsonl`、`AGENT_REVIEW_NOTES.md` | 失败、30例待人工复核包、Agent定性检查 |
| `final-v2-tests.xml`、`final-v2-tests.log` | 冻结时102项测试结果 |
| `live/`、`integration/`、`reflect-diagnostic/` | 中断/初次运行；保留成本和错误，不能混入最终分数 |

每请求JSON与汇总JSONL同时保留：前者支持恢复和成本审计，后者便于复算。数据库/向量运行目录、凭据不提交。`changed-files.txt`为实验结束时快照，不是当前发布清单。
