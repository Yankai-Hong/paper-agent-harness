# RAG 与混合检索

实现：[retrieval.py](../src/paper_agent/retrieval.py)。RAG 是先检索本地证据，再让生成器依据这些证据回答；本组件承担检索部分。

| 环节 | 实现与来源 | 默认值 |
|---|---|---|
| 稀疏召回 | [rank_bm25](https://github.com/dorianbrown/rank_bm25) BM25Okapi | 30 候选 |
| 稠密编码 | [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) BGEM3FlagModel | 归一化 float32 向量 |
| 向量索引 | [FAISS](https://github.com/facebookresearch/faiss) IndexFlatIP | 精确内积 |
| 融合 | RRF；参考 [Pyserini](https://github.com/castorini/pyserini) rank fusion | k=60 |
| 最终证据 | 混合候选后可选 Cross-Encoder | 30 进、5 出 |

RRF 对同一 chunk 在各路的排名求和：`sum(1 / (60 + rank))`，rank 从 1 开始。它不直接相加 BM25 分数和向量分数；重复 ID 去重，平分按 chunk_id 排序，保证可重放。

`search(query, mode, top_k=5, candidate_k=30, paper_ids=None)` 支持 bm25、dense、hybrid、rerank。`paper_ids=None` 搜全库，空数组检索空范围，指定 ID 则只搜索相关论文。先过滤论文范围，再截取每路候选，避免全库前 30 把目标论文挤出。

当前约 2,000 chunks，FAISS 对全索引精确搜索后过滤足够实用；不是面向百万文档的服务。编码和重排可在 GPU，FAISS 安装的是 faiss-cpu，索引查询在 CPU。不要将整个检索栈称为 GPU FAISS。

索引保存 corpus hash、模型来源、最大长度、FAISS 文件 SHA256。加载时校验，缺模型或不匹配时失败，不自动替换成随机向量/小模型。`retrieve_chunk(chunk_id, neighbor_window=0..2)` 提供同论文邻块读取；目前七阶段并未自动调用邻块扩展。

BM25 使用 lexical-v1，适配本次英文论文；中文切词不是本次已评测能力。BGE 使用 dense 输出，没有额外启用其 learned sparse 或 ColBERT 分支。

实测与限制见 [RESULTS](../reports/history/development/qasper-suite/RESULTS.md)：主指标是已知论文范围的段落覆盖 Recall@5，不是开放域问答准确率。

## 本轮可靠性与复测

独立检索评测仍严格失败，不把降级结果冒充指定算法。Pipeline 在 allow_degraded=True 时，hybrid 失败依次尝试 BM25/dense，成功记录 fallback 与 degraded；两路均失败则失败并回滚。当前四组结果见 reports/final-acceptance/retrieval-reused；历史排名和区间见 reports/history/development/acceptance/retrieval-v2。Recall@10、MRR 和二元相关 chunk nDCG 口径见 [评测](evaluation.md)。
