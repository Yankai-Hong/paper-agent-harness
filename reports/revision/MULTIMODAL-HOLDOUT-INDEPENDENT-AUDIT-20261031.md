# 论文隔离保留集：独立 agent 原页复核

复核对象是冻结前未读取检索结果或模型输出的 `data/eval/multimodal-holdout-v1/cases.json`，本次核对的 SHA-256 为 `26f06df36b6b7a06f1c6456d2d96b1db1ab61ee2820f0526bb21e64bf9c56d76`。本 agent 未参与选题，对照两篇原 PDF、六张原页渲染、三张查询图及标注独立检查；能看到项目上下文与题集说明，因此不是盲审或人类金标。**未查看任何检索或生成结果，也未调用模型。** 冻结题和原标注未修改。

## 来源、范围与限制

原 PDF 是 [Deep Contextualized Word Representations](https://aclanthology.org/N18-1202/)（N18-1202，11 个物理页，SHA-256 `7539c5dbcabadd66a3096dd59cb4275e5ce1c44a1161d5775cea38bdd0914e23`）和 [Sentence-BERT](https://aclanthology.org/D19-1410/)（D19-1410，11 个物理页，SHA-256 `e2703128039850ba7e098bfc6c7d9f3f3e75e6318815c5dd90af71085b1d384e`）。哈希与题集一致。原件、页图和查询图片在被忽略的 `data/raw/multimodal-holdout-20261031/`；本报告不再分发图片或原文。

[ACL 官方版权说明](https://aclanthology.org/faq/copyright/)将 2016 年起的 ACL 材料置于 CC BY 4.0，并明确不覆盖第三方材料。该政策支持题集记录的来源条件，但不能单凭这一政策独立证明论文中所有可能的第三方材料均可再分发；本任务维持本地留存和必要上下文的既有授权。

## 固定问答逐例

页码是 PDF 从 1 开始的物理页；下述 `pages/...` 均相对于 `data/raw/multimodal-holdout-20261031/`。图表目标的 `bbox_points_approx` 是标注者近似范围，不能作为像素级精确真值。

| ID | 原页核查及与冻结标注的关系 |
|---|---|
| h01 | `pages/D19-1410-p3.png`：Figure 1 图注说两路 BERT 权重绑定；图和右栏分类目标均显示将 u、v、逐元素绝对差送往分类器。标注支持。 |
| h02 | 同页 Figure 2 图中，两路 pooling 后的 u、v 进入 cosine-sim；右栏 regression objective 同样说明余弦相似度。标注支持。 |
| h03 | `pages/N18-1202-p8.png`：Figure 1 左面板是 SNLI/Accuracy、右面板是 SRL/F1。标注支持。 |
| h04 | `pages/N18-1202-p5.png`：Table 1 的 SRL 行 baseline 为 81.4、ELMo + baseline 为 84.6；表注指定 SRL 度量为 F1。标注支持。 |
| h05 | `pages/D19-1410-p4.png`：Table 1 Avg. 列 SBERT-NLI-base 为 74.89、large 为 76.55；表注说明 Spearman ρ × 100。标注支持。 |
| h06 | `pages/N18-1202-p1.png`：摘要和右栏分别说明双向语言模型内部状态及 ELMo 名称全称。标注支持。 |
| h07 | `pages/D19-1410-p1.png` 摘要报告，在 **10,000 个句子中找最相似的一对**时，BERT 约 65 小时、SBERT 约 5 秒。冻结 JSON 的问题正确；README 中“10,000 句子相似对”的中文表述易被误读为仅 10,000 对，应解释为约 5,000 万次成对推理，且不是本机性能。 |
| h08 | `pages/N18-1202-p8.png` Figure 1 及 Sample efficiency 正文均显示 SRL 相关比较；正文明确称 ELMo 1% 与 baseline 10% 的 F1 约相当。标注支持。 |
| h09 | `pages/N18-1202-p1.png` 摘要描述依上下文变化的词表示，`pages/D19-1410-p1.png` 摘要描述可比较的句子嵌入；两个来源均为必要依据。标注支持。 |
| h10 | 两篇原文均有不同的 Figure 1（N18 第 8 页、D19 第 3 页）。在无历史上下文、未限定论文的独立会话中应澄清；不能给单一命中金标。若存在真实历史指代，结论可能变化。 |
| h11 | D19 论文中模型权重文件的 SHA-256 **没有被本次核查找到**，保守结论为 unknown。11 页 PDF 文字层对 `SHA-256`/`SHA256` 的字面计数为 0，但这不证明所有图片、附件、外部文件或哈希不存在。 |
| h12 | `pages/D19-1410-p3.png` Figure 2 是双 BERT 池化后余弦；`pages/N18-1202-p8.png` Figure 2 是 biLM 层权重图。冻结的目标和对照来源均支持区分两篇。 |
| h13 | `pages/D19-1410-p3.png` Figure 1 是句向量 u、v 与绝对差进入 softmax 的图。该标注在**本冻结的两篇论文范围内**有充分依据；不声称整个世界或任意扩展论文库中该描述唯一。 |

`h01`/`h08` 的图和正文需要分别核对，`h09` 两篇来源均必要；`h10` 不纳入唯一目标召回分母，`h11` 不把缺乏可核实答案记为漏检。原始输出尚未产生，本报告对 Top-5、引用及逐声明支持均不预先赋值。

## 三个独立查询图

查询图文件 SHA-256 均与冻结 JSON 一致。分别用冻结的 100 dpi 原页 ROI 与 Lanczos 缩放重新计算，三个查询图的尺寸及 RGB 像素均与已存查询图相同；这仅验证查询图来源与变换，不证明任何系统返回引用正确。

| ID | agent 核查 |
|---|---|
| himg01 | D19 第 3 页 Figure 1 的分类结构，目标图和 query 不同文件；Figure 2 同页同为双 BERT 结构，可作为相似干扰。 |
| himg02 | D19 第 3 页 Figure 2 的 cosine 双路结构；已存查询图**右侧切去部分 v/右路框线**，仍能识别余弦和双路结构，是有效的局部裁剪查询，不能当作完整图或引用证据。Figure 1 是相似干扰。 |
| himg03 | N18 第 8 页 Figure 1 两个训练数据曲线面板，SNLI/accuracy 和 SRL/F1 均在查询图中。其他图表保留为潜在干扰，具体检索排序未执行、未知。 |

所有原页已视觉查看；没有由目标坐标猜测系统检索输出，没有修改固定题、答案或图号。后续的候选命中、页码区域、逐声明语义支持、误拒及模型用量须根据真实运行另记，不可将本标注复核当作系统验收。
