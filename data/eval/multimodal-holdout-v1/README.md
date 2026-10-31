# 多模态论文隔离保留集 v1

固定文件 [cases.json](cases.json) 的 SHA-256 为 `26f06df36b6b7a06f1c6456d2d96b1db1ab61ee2820f0526bb21e64bf9c56d76`。选题、答案、图号和目标区域在查看项目检索输出或请求模型之前固定。标注者是 agent，不是人类；独立 agent 复核若进行，另记身份、分歧与 unknown。不要因系统结果更改本文件的题目或金标：若用其迭代实现或阈值，应将其降为开发集，再另建未用于调参的保留集。

两篇论文与此前五篇多模态开发论文不同；本地 QASPER `papers.jsonl` 中按论文 ID 和完整标题检查均无相同条目。PDF 原件、页渲染及查询图仅放在被忽略的 `data/raw/multimodal-holdout-20261031/`，不提交仓库。物理页从 1 开始，区别于论文印刷页码。字段 `bbox_points_approx` 是根据原页目视确定的包围区域，坐标为 PDF points、左上原点，**用于辅助审阅而非像素级真值**；检索定位不确定时应另行复核或标 unknown。

| paper_id | 官方来源与版本 | 原 PDF SHA-256 | 条件 |
|---|---|---|---|
| N18-1202 | [Deep Contextualized Word Representations, NAACL 2018](https://aclanthology.org/N18-1202/)；[原 PDF](https://aclanthology.org/N18-1202.pdf) | `7539c5dbcabadd66a3096dd59cb4275e5ce1c44a1161d5775cea38bdd0914e23` | PDF 页首标 ACL 2018 copyright，适用下述 ACL 2016 年后政策。 |
| D19-1410 | [Sentence-BERT, EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1410/)；[原 PDF](https://aclanthology.org/D19-1410.pdf) | `e2703128039850ba7e098bfc6c7d9f3f3e75e6318815c5dd90af71085b1d384e` | PDF 页首标 ACL 2019 copyright，适用下述 ACL 2016 年后政策。 |

[ACL 官方版权 FAQ](https://aclanthology.org/faq/copyright/)说明 2016 年及以后的 ACL 材料使用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；第三方材料不自动受该政策覆盖。论文原件和图表不发布到仓库，来源和作者应保留署名。本次仅按用户授权向现有 OpenCode Go 模型发送必要的公开论文图表与上下文，不向遥测服务发送内容。

## 金标与核查路径

本地原页 `data/raw/multimodal-holdout-20261031/pages/<paper_id>-p<page>.png` 由 Poppler `pdftoppm -r 100 -png -f <page> -l <page> -singlefile` 渲染，留供 agent 比对。下列答案均由本 agent 对照原页和相应正文确定，而非模型输出：

| ID | 范围 / 类型 | 原页定位与核查依据 |
|---|---|---|
| h01 | 明确单篇；图+正文 | D19-1410 p3 Figure 1、Classification Objective Function：两个 BERT 权重共享，分类器输入为 u、v 与逐元素绝对差。 |
| h02 | 明确单篇；图 | D19-1410 p3 Figure 2：池化后用 cosine-sim(u,v)。 |
| h03 | 明确单篇；曲线 | N18-1202 p8 Figure 1：SNLI/accuracy 与 SRL/F1 两个面板。 |
| h04 | 明确单篇；表 | N18-1202 p5 Table 1 的 SRL 行，baseline 81.4，ELMo+baseline 84.6，指标 F1。 |
| h05 | 明确单篇；表 | D19-1410 p4 Table 1 的 Avg. 列，SBERT-NLI-base 74.89、large 76.55；表注为 Spearman 相关系数乘 100。 |
| h06 | 明确单篇；正文 | N18-1202 p1 Abstract/Introduction：ELMo 名称与双向语言模型内部状态。 |
| h07 | 明确单篇；正文 | D19-1410 p1 Abstract：论文从 10,000 个句子中寻找最相似的一对，报告 BERT 约 65 小时、SBERT 约 5 秒；不当成本机计时。 |
| h08 | 明确单篇；图+正文 | N18-1202 p8 Figure 1 与 Sample efficiency：SRL 上 ELMo 1% 训练数据约等于 baseline 10% 的 F1。 |
| h09 | 明确跨两篇；正文 | N18-1202 p1 ELMo 是上下文相关词元表示；D19-1410 p1 SBERT 是句子嵌入，两篇均为必要证据。 |
| h10 | 全库歧义 | 无论文上下文的“图 1”同时可指 N18-1202 p8 和 D19-1410 p3；应澄清，不把任一个当唯一金标。 |
| h11 | 明确单篇；证据不足 | 权重文件 SHA-256 未有核对依据，期望说明无法确认；这不是对外部文件或所有隐藏材料不存在的证明。 |
| h12 | 全库、上下文明确；图 | 两篇标题均出现，D19-1410 p3 Figure 2 双 BERT 池化后余弦；对照 N18-1202 p8 Figure 2 为层权重热图。 |
| h13 | 全库、上下文明确；图 | D19-1410 p3 Figure 1 的双句向量、绝对差与 softmax 足以排除仅按“图 1”猜论文。 |

三个图搜图查询均独立从原 PDF 页渲染裁剪、缩放，不是来自 Docling 索引或返回的证据文件。D19-1410 的 Figure 1 和 Figure 2 位于同页且均为双 BERT 架构，互为相似干扰；N18-1202 Figure 1 为两联训练数据曲线。

| ID | 查询图 | 原页像素 ROI（100 dpi） / Lanczos 缩放 | SHA-256 |
|---|---|---|---|
| himg01 | `query-himg01.png` | D19-1410 p3 `[140,85,360,327]` / 0.8 | `2235cf1cd54ede18ed7b989008f63ef395d2c97c677887e37a839e5685cad7c1` |
| himg02 | `query-himg02.png` | D19-1410 p3 `[468,85,665,327]` / 1.15 | `8a0e27c6a0aec27ec01c5a4ed385e76d5e387798da58239bdd9a091178689e9b` |
| himg03 | `query-himg03.png` | N18-1202 p8 `[430,125,705,300]` / 1.1 | `fa5b2583265e8f1049121e6581be799d40946b2bac8245adda7d9a051eaa9dd2` |

## 评测口径与分母

固定 **13 个问答 + 3 个图搜图**：h01–h08、h11 为明确单篇（9）；h09 明确跨两篇（1）；h12–h13 为全库但上下文足以定位（2）；h10 为全库歧义、期望澄清（1）。h11 的唯一可核查结论是证据不足/unknown，不得将全文字面搜索缺失当作外部文件不存在的证明。

报告明确范围、无范围、跨论文和歧义各自分母；h10 不给唯一目标，也不纳入唯一答案 Recall 分母；h11 不把无目标误记为检索失败。图/表任务按可信原页、图号、区域检查；正文任务必须核对实际段落而非仅同页。h01 和 h08 需要图及正文均覆盖，h09 需要两篇均覆盖；分别报告每证据和完整证据召回。结构合法、页码/区域正确及逐声明语义支持分别计数；未执行、解析漏检、不可辨和分歧保持 unknown，禁止静默换目标。

顶级 JSON 键为 `version`、`frozen_before_holdout_execution`、`label_status`、`paper_isolation`、`papers`、`cases` 和 `image_queries`。`cases` 每项含 `id`、`kind`、`scope`、`paper_ids`（`null` 表示全库）、`question`、`expected` 和 `evidence`；歧义项另有 `acceptable_targets`。每个证据含 `paper_id`、物理 `page`、`type` 和 `locator`；图表可带近似 `bbox_points_approx`。`image_queries` 带目标论文、页、图号、原页裁剪参数、查询图哈希及干扰说明。
