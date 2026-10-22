# 组件文档

建议先读架构，再依次读数据、检索、路由、生成、记忆与评测。所有实现入口均指向当前代码；来源和许可证集中在 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。

| 文档 | 内容 | 实现 |
|---|---|---|
| [实现审计](implementation-audit.md) | 七阶段逐项核对、缺陷与修改建议 | 全源码 |
| [架构与七阶段](architecture.md) | 状态、流程、重试、职责边界 | pipeline.py |
| [语料与分块](corpus.md) | PDF/JSONL、token 窗口、元信息 | corpus.py |
| [RAG 与检索](rag-retrieval.md) | BM25、BGE-M3、FAISS、RRF | retrieval.py |
| [重排序](reranking.md) | Cross-Encoder、候选预算、GPU | retrieval.py |
| [路由与 Query 改写](routing.md) | 五类意图、指代、论文范围、澄清 | providers.py / pipeline.py |
| [证据反思](reflection.md) | 证据充分性、二次检索、拒答 | pipeline.py |
| [生成与引用](generation.md) | 结构化事实、引用校验、限制 | providers.py / memory.py |
| [记忆机制](memory.md) | 会话恢复、事实更新、压缩、归档 | memory.py / pipeline.py |
| [工具循环与追踪](runtime-tracing.md) | learn-claude-code 复用、trace、usage | runtime.py / pipeline.py |
| [API 配置](api.md) | OpenCode 实测、超时重试、预算与费用边界 | providers.py |
| [评测与复现](evaluation.md) | GPU 实测、eval 子命令、Judge、消融 | scripts/ / eval_cli.py / judge.py |

当前验收记录：[最终验收](../reports/FINAL_REPORT.md)、[真实检索结果](../reports/qasper-suite/RESULTS.md)、[真实 API 评测](../reports/LIVE_EVAL_RESULTS.md)。

配套报告：[消融](../reports/ABLATION_RESULTS.md)、[记忆](../reports/MEMORY_EVAL_RESULTS.md)、[可靠性](../reports/RELIABILITY_RESULTS.md)。未完成项清单与复现计划为内部记录，未随仓库发布。
