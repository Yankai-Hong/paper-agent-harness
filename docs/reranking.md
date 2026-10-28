# Cross-Encoder 重排序

实现：[BGEReranker](../src/paper_agent/retrieval.py)，调用官方 FlagEmbedding 的 FlagReranker。模型：[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)。

双编码器可以预先编码全文；Cross-Encoder 则把问题与每个候选原文组成一对共同计算相关性，因此只对融合后的前 30 个候选运行。使用原始分数排序，取前 5；分数不是答案正确概率。

本机使用 cuda:0、FP16、batch_size=4、max_length=768。BGE 分块长度为 500，仍需留意很长的问题可能触发 pair 截断。输出长度/数值有效性在检索组件检查；严格检索评测中模型缺失会报错；Pipeline 的显式降级另见下文。

固定 revision：`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`。下载来源及权重 SHA256 记录在 `models/bge-reranker-v2-m3/provenance.json`，评测另存副本。Hugging Face 下载受阻时可从 BAAI 官方 ModelScope 渠道下载，但启用前必须匹配固定权重哈希。

本次重排 Recall@5=77.61%，比 hybrid 提升 3.38 个百分点。按论文聚类的配对 bootstrap 区间跨越零，不据此声称稳定显著提升。完整设置与预测见 [评测报告](../reports/history/development/qasper-suite/RESULTS.md)。

## Pipeline 与严格检索评测的区别

Pipeline 默认 allow_degraded=True：Cross-Encoder 不可用/分数非法时保留原检索排序，记录 degraded=true 和 reranker_unavailable。eval 的 Pipeline 构造可容忍加载失败，独立四组检索评测仍严格报错；不能把降级输出计为真实 Cross-Encoder 排序成功。
