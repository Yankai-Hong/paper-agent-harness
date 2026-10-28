# 无需 API 的实施记录

日期：2026-10-22。用户要求优先完成无需付费 API 的部分，并使用 GPU 环境；随后要求补齐兼容 API、README 和组件文档。下方早期记录已更新到当前状态，没有付费 API 调用。

## 已完成并验证

- 在 Conda minimind 的 GPU PyTorch 基础上创建项目 `.venv-gpu`；Python 3.10、PyTorch 2.11.0+cu128、RTX 4060 Laptop。环境依赖校验无冲突，版本闭包见 requirements-lock.txt。
- 完成真实 QASPER 数据下载与固定抽样：100 篇全文、100 个官方证据检索问题。仅正文/摘要入库，不把问题答案标签入库。
- 完成 JSONL/PDF 文本导入、段落分块、原文字符范围、BGE tokenizer 滑窗与稳定 chunk ID；BGE 分词配置生成 1,996 chunks。
- 完成 rank_bm25、FAISS 精确索引、RRF、BGE-M3/重排序模型适配器和 CLI；缺少真实模型时明确失败，不做隐式替换。
- 完成基于 learn-claude-code s02 的有限工具循环：schema 检查、tool_use/result 配对、错误反馈和 usage 汇总。
- 完成七阶段领域编排、最多两次检索、结构引用校验、会话事实与指代存储、结果归档和 trace。模型决策通过注入接口提供，当前使用脚本响应测试，未调用真实 Claude。
- 完成离线指标函数：检索召回、意图分类、边界决策、拒答 precision/recall、误拒答率、人工判断输入的 pipeline success、记忆检查点和含缓存字段的 token 对比。
- 自动化测试见 implementation-tests.xml。包含真实 FAISS/rank_bm25、上游压缩兼容、本地模拟 HTTP 的完整问答链路、会话恢复、批次限额和评分约束；fixture 仅用于逻辑验证，不用于效果报告。
- 已接入 DeepSeek / OpenAI-compatible Chat Completions，提供 ask/chat 和批量多轮运行。配置读取 .env，校验模型 JSON，记录调用与缓存 usage，无自动重试；尚未真实供应商验收。
- 已生成 30×4 来源可追溯对话草稿及 100/55/70 视图与审阅模板，全部标为未复核；评分器拒绝草稿标签。README 和 docs 各组件说明已补齐。

## 已测基线

| 数据/配置 | Recall@5 | Hit@5 | 失败/缺失 |
|---|---:|---:|---:|
| 主评测：QASPER 100题，已知论文范围，BGE 分词 500/80，BM25 | 66.26% | 68.00% | 0/0 |
| 同设置：dense | 73.94% | 78.00% | 0/0 |
| 同设置：hybrid | 74.24% | 77.00% | 0/0 |
| 同设置：rerank | 77.61% | 81.00% | 0/0 |
| 压力测试：同100题，不提供论文范围，BGE 分词 500/80，BM25 | 24.33% | 25.00% | 0/0 |

主评测见 qasper-bm25-scoped/；全库压力测试见 qasper-bm25-bgechunks/。词法分块 smoke 基线另存 qasper-bm25-lexical/，不与不同 chunk 设置混做消融。

口径修订记录：首次先跑了全库版本；核查 QASPER 的论文上下文假设后，在正式 dense/hybrid/rerank 运行前增加已知论文范围的主视图，未改变样本或检索超参数。保留原全库结果与原查询视图 retrieval-global.jsonl，不能把任务范围改变解释成算法提升。

注意：这是英文单轮检索基线。段落覆盖以任一对应窗口命中计分，是代理指标，不保证完整支持答案。尚无与截图 7.1/3.2 个百分点提升直接可比的结果。

## 模型与 GPU 实测状态

完整权重已下载、验哈希，四组 GPU 实测已完成，见 [真实结果](qasper-suite/RESULTS.md)。hybrid 相对 dense +0.30pts，rerank 相对 hybrid +3.38pts，配对 bootstrap 区间均跨零。GPU 峰值 allocated 为 2,333,472,256 字节；FAISS 索引在 CPU，编码与重排在 GPU。

首次 suite 跑完前三组时重排权重尚未就绪，随后复用同一索引重跑四组成功。报告保留最终四组成功结果及其运行记录。单次耗时受冷暖状态影响，不据此做吞吐结论。

下载使用 BAAI 官方模型 ID。Hugging Face 大文件传输超时后，改从 BAAI ModelScope 发布页分段下载；全部权重必须通过固定 Hugging Face revision 的 SHA256 校验后才启用。下载过程没有模型推理 API 费用。

## 后续需要真实模型或人工参与的部分

- DeepSeek/兼容 API 的真实联通与 NLU、Reflect、生成效果：尚未使用真实密钥验证。Go 面向编程流量，论文问答需确认适用性，默认推荐 DeepSeek。
- 30×4 草稿的自然性、gold 证据与意图/边界标签需要独立人工复核。100 条官方检索集另行取自 QASPER，不能声称都是从这 120 turns 划分得到。
- 当前工作记忆自动压缩是裁剪与可回读归档，没有自然语言摘要质量结论；短四轮对话可能不触发压缩，需长会话压力集。
- 引用语义支持、正确拒答、真实记忆压缩后的问答一致性和 token 降幅：需真实生成及独立判断。现有引用验证只保证 ID 存在及论文一致。
- 截图中的指标全部保持“待复测”，不能用本次测试通过率代替。

最终 29 项测试通过；文档链接、四组保存预测的指标重算和草稿状态均通过检查，见 [delivery-verification.json](delivery-verification.json)。本轮没有付费 API 调用。实际运行步骤见根目录 README。
