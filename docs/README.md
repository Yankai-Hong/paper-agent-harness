# 文档导航

当前结论以 [最终工程验收](../reports/FINAL_ACCEPTANCE.md) 为准。源码、运行配置与评测证据一致；结果为工程验收和模型辅助评审，不是人工金标或论文级增益证明。

| 文档 | 内容 |
|---|---|
| [安装与运行](setup.md) | 环境、模型、数据、索引、API 与完整三层问答 |
| [架构](architecture.md) | 七阶段、状态、分支与实现边界 |
| [当前验收审计](implementation-audit.md) | 修复、验证与保留限制 |
| [语料](corpus.md) | PDF/JSONL、段落与滑窗、来源元数据 |
| [检索](rag-retrieval.md) / [重排](reranking.md) | BGE、FAISS、BM25、RRF、Cross-Encoder |
| [路由](routing.md) / [反思](reflection.md) / [生成](generation.md) | NLU、证据充分性、引用与拒答 |
| [记忆](memory.md) | Redis/SQLite/Chroma、版本、提交隔离与恢复 |
| [追踪](runtime-tracing.md) / [API](api.md) | Trace、请求预算、有限重试与费用边界 |
| [评测](evaluation.md) | 当前配置、分母、复现与结果解释 |

历史实验统一见 [报告导航](../reports/README.md)，历史分数不替代当前成绩。上游来源与许可见 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。
