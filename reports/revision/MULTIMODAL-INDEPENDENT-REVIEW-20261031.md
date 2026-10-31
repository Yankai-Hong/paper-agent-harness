# 固定多模态案例独立 agent 复核

本记录由另一 agent 对照固定题集、五篇公开 PDF 的原页、原始首轮及后续预测、对应 v3/v4 证据目录和检索记录独立检查；该 agent 可看到项目和此前工作记录，复核不是盲审，也不是人类审查或人类金标。定位仍无法确认的项保留 unknown。全部 PDF、原页、查询图、结果和裁剪均为本地受忽略资产，不随报告再分发。物理页码从 1 开始。

首轮使用 `indexes/multimodal-v3-reviewed` 与 `runs/multimodal-acceptance-20261031/predictions.json`；c02、m01 关闭思考模式的针对性复验使用 v4 与 `runs/multimodal-followup-20261031/predictions.json`。从各自目录共核对 28 条引用：键、证据类型、页码、bbox 均与对应证据一致，视觉引用路径和文件哈希均匹配。该检查不证明语义支持；下文逐条判断。

## 固定问答

1. **t01 / 1903.10318 p1**：原页 Methodology 定义每句二元标签为是否进入摘要，金标成立。v3 融合 Top-5 第 5 位为 `v-50fe5151b82a14b7aa41ba7a`。唯一回答声明与该段及 bbox 对应，支持；无图表引用。原页：`data/raw/multimodal-eval/review-20261031/1903.10318-p1.png`。
2. **t02 / 1909.12079 p3**：原页左栏 KB Type Representation 明确使用 Freebase，金标成立。首轮未检索而澄清，目标 Top-5 unknown，未作答，不评价引用/支持。原页：`data/raw/multimodal-eval/check-1909.12079-p3.png`。
3. **t03 / 2002.06854 p1、p3**：p1 摘要约 5000 万条评论，金标成立；首轮 Top-5 不含指定摘要，p3 的 `v-7b4f5270cef04e307a1645d5` 是第 4 位并独立支持同一数字。唯一回答和引用位置支持。原页：`data/raw/multimodal-eval/check-2002.06854-p1.png`、`data/raw/multimodal-eval/review-20261031/2002.06854-p3.png`。
4. **a01 / 1909.12079 p3 Figure 1**：图 1 第一、正文第二进入 Top-5；图中三层 MLP 和正文 `v-487a30a56ee52b65eee288cf` 支持三个声明：拼接后进入 MLP、三层、输入含链接置信度 g。图 `v-1c73e3eb3bf5909f220f8283` 与原页一致；图表引用和正文坐标相符。原页 `check-1909.12079-p3.png`。
5. **a02 / 1911.04474 p4 Figure 2**：图 `v-9edbf9abcc28c690822ef3b2` 第 1，图注 `v-1a31766cb10d4e1ab0ccacf1` 第 2。四项声明分别为 embedding 输入及融合、Transformer 上下文、CRF 标签和图注说明，图及图注支持；引用对应。原页 `check-1911.04474-p4.png`。
6. **a03 / 1909.06708 p4 Figure 3**：图 `v-f1f21457eabaec0c5e61d692` 第 1。绿色 hidden-state hints 与黄色 word-alignment hints 均可从图例和箭头辨认，两声明及引用支持。原页 `check-1909.06708-p4.png`。
7. **c01 / 1909.12079 p5 Table 1**：表 `v-29e405dbc6e7a1ebeb55ada0` 第 1，原表 FIGER (GOLD) accuracy 一列分别给出 Ours (Full) 75.5 和 Ours (NoEL) 69.8；原页 Results 同列称其为 strict accuracy。两声明、原表位置及引用支持。原页 `check-1909.12079-p5.png`。
8. **c02 / 2002.06854 p4 Figure 2**：原页图注关联四个面板，固定金标四项统计成立。首轮 v3 Rerank Trace 中复合图 `v-53ad4074b40eca71d6882c6a` 第 1，Generate 截断且无可核查回答或引用。v4 复验同图第 1；五项声明分别描述每用户评论（2.24±3.95）、每酒店评论（137.69±342.94）、每条评论词数（125.57±128.91）、每年评论和图注所称对数刻度，均有图及图注依据；中文“字数”易误解为字符数，括号英文明确 words，应称词数。金标的“坐标使用对数尺度”也应精确为图中对数刻度：面板 (d) 的横轴年份为线性，不能声称每一坐标轴都是对数。四个原面板及子图标题均在复合裁剪中可见；原页 `check-2002.06854-p4.png`。
9. **m01 / 1903.10318 p2 Figure 1 和正文**：原页显示每句前 CLS、句后 SEP、顶层 CLS 向量作句表示、奇偶句 EA/EB 交替，金标成立。首轮 v3 Rerank 图 `v-f74e776a25cbd737711fcd3b` 第 1，指定 p2 正文段 `v-1d6d22fef1dc6123f414cb06` 和 `v-65035b9e45cfec593dae84de` 均不在 Top-5；Generate 截断无可核查回答。v4 复验图第 1，引用正文为 p1 `v-69140eaebfebdcf52c011bd3` 和 p2 `v-6469e879dd682f5ff372ced7`，并非上述两个固定目标正文段。复验前五声明（BERT 词元输出及修改、CLS/SEP、EA/EB、两类原有段标签、位置嵌入）分别有图和相应正文支持；第六声明称最终输出 Y1/Y2/Y3 是句子表示，原页 2.2 将输出定义为预测分数，故该附加声明错误，其引用不能支持；整答不应评为完全正确或完全受支持。原页 `check-1903.10318-p2.png`。
10. **u01 / 1909.12079**：固定问题询问权重文件 SHA-256。首轮检索前拒答，Top-5、引用、证据反思均 unknown；六页文字层没有字面哈希不证明不存在外部权重或哈希。拒答形式符合保守方向，但不能判定系统已基于检索证据正确拒答。定位为上述论文；无可指认的回答证据区域。

## 图搜图与版本差异

查询图均从原页局部裁剪再缩放，不是作为引用返回的可信证据图。原始 `predictions.json` 与 `output/clip-retrieval-v3-reviewed.json`、v4 等价搜索记录都显示目标第 1：

| 案例 | 查询图 | 目标实体与原页 | Top-5 中的相似干扰 |
|---|---|---|---|
| i01 | `data/raw/multimodal-eval/query-i01.png` | 1909.12079 p3 Figure 1；`v-1c73e3eb3bf5909f220f8283` | 1903.10318 p2 Figure 1 第 2、1911.04474 p4 Figure 2 第 5，均为网络结构图。 |
| i02 | `data/raw/multimodal-eval/query-i02.png` | 1911.04474 p4 Figure 2；`v-9edbf9abcc28c690822ef3b2` | 1903.10318 p2 Figure 1 第 2、1909.12079 p3 Figure 1 第 3、1909.06708 p4 Figure 3 第 5。 |
| i03 | `data/raw/multimodal-eval/query-i03.png` | 1909.06708 p4 Figure 3；`v-f1f21457eabaec0c5e61d692` | 1903.10318 p2 Figure 1 第 2、1911.04474 p4 Figure 2 第 4。 |

我将三个查询图与上述原页及检索到的目标裁剪作视觉对照；它们是不同论文的架构图，具有节点/箭头相似性，不能仅凭相似外观当作同一证据。目标均对应所列页和编号；3/3 是极小样本，不推断普遍性能。

`2002.06854` p4 的 Figure 2 四块原图、标签及图注均在原页；v3/v4 的组合区域是按已登记的人工指定 bbox 合并，不是 Docling 自动完整检出。`1909.06708` p3 Figure 2 原页有三面板：v3 错将整张 Figure 2 图注附给单面板 `v-1010a8202ecfa541d6be73dd`；v4 将此条排除，其他无编号面板仍存在，v4 并未拥有完整 Figure 2 复合图。`2002.06854` p1 Figure 1 是整条评论，原误关联仅为评分图标，v3 已排除；Table 1 仍可能编号缺失。新增独立渲染：`data/raw/multimodal-eval/independent-review-20261031/1909.06708-p3.png` 与 `2002.06854-p1.png`。

对六个固定图表目标，重新按 PDF 物理页、bbox 和 scale=2 渲染：六者尺寸均一致且所存文件 SHA-256 匹配；HotelRec 人工复合图像素逐点一致，其余五个 Docling 裁剪与单独 PDFium 渲染并非像素逐点一致，RGB 相关系数约 0.949–0.997，视觉与页区域一致，不可声称像素完全一致。截断、遗漏和错误关联不因哈希匹配而自动排除。
