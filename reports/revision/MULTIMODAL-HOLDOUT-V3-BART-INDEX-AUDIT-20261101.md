# V3 BART 单篇解析：目标证据原页核查

本报告由未参与实现和模型调用的独立 agent 对照 BART 原 PDF、冻结题集、`data/raw/multimodal-holdout-v3-bart-only/evidence.jsonl`、`text.jsonl` 与 `data/raw/multimodal-holdout-v3-docling/crops/` 完成；**不是人工复核或盲审**。未查看本集模型输出、未改动代码、题集或索引。该证据集只包含 BART 一篇，ULMFiT 在冻结版解析时失败，**不是完整两篇语料或两论文保留集的验收**。下述为原页/解析实体覆盖，不预设检索排名，也不将题集目标冒充命中。

页数是 PDF 物理页，bbox 为左上原点 points，以下四舍五入显示；来源 PDF SHA-256 均为已核对的 `3562c723fe7bebb8b362533122b3ca5c065037c8e8d67460654cf3740652bb59`。

## v302：Figure 2 与 Text Infilling 正文

- 图 `v-a4e9cf505dc170fd9903bd70`：第 3 页，bbox `[140.95,61.05,454.13,146.41]`，`figure_number=Figure 2`。实际 crop 包含 token masking、sentence permutation、document rotation、token deletion、text infilling 五个示意和标签，整体对照原页完整；Text Infilling 位置正确。图框不包含下方独立图注或 λ 数值。
- 同页独立图注正文 `v-1faf7ec1d547486b39cb79a1`，bbox `[72.00,161.05,525.55,169.61]`。图像元数据关联状态为 `docling_link_unverified`；本具体页已人工之外的 agent 对照原图核查，不能推广其他图注关联。
- 同页 Text Infilling 段落 `v-6eb2eb027e9afb995f1da312`，bbox `[72.00,224.64,290.27,353.14]`，包含每个 span 使用一个 `[MASK]`，以及 Poisson 分布 λ=3；正文也进入合并 `text.jsonl`。图本身不能证明 λ=3 或把正文直接当成框内内容。
- 图文件 SHA-256 实测为 `81410eef3f6af79cb5ba46b798e7c79b6e8907d31c92c484e5f179b363933a7b`，与记录一致；哈希匹配不单独证明区域语义。

## v303：Table 1 表格与图注

- 第 5 页表格裁图 `v-7e971512098372bc26f89a4c`，bbox `[79.11,62.63,517.70,263.22]`；对照原页，表框覆盖所需 BART Base、XSum PPL 列及 Text Infilling 6.61、Text Infilling + Sentence Shuffling 6.62 两行。图像本身包含 XSum 和 PPL 列标题及行名，所需数值和单位可直接目视核查。
- 该图像实体的 `figure_number=null` 且 caption 为空；独立文字 `v-00f66a7a9464f883e5e2489f` 在同页 bbox `[72.00,280.07,525.55,348.39]`，内容为 Table 1 图注，并已进入 `text.jsonl`。对照原页该图注属于上方表格，但本次**不修改**未识别的图号或关联。精确按 `Table 1` 编号过滤时该实体可能被排除；是否发生应以实际过滤运行记录为准，不能事先假定。
- 表图文件 SHA-256 实测为 `c258b3da4997fe6d67bc9dc6ed0208f5e45817fb4e4e896e804a42027060e9c9`，与记录一致。表格抽取结构、未提及的单元格及其他跨页/错位未穷尽核查。

## v308：完整标题与首位署名作者

原 PDF 第 1 页的标题和署名是两个独立正文实体，并且都出现在 `text.jsonl`：

- 标题 `v-8b5eae33d2488c0053126e62`，`docling_label=section_header`，bbox `[99.48,71.79,498.06,100.63]`；文本与原页完整 BART 标题一致。
- 署名 `v-1ddb443c2b36c90376d92bc3`，`docling_label=text`，bbox `[118.99,127.20,478.56,165.82]`；原页及块中 Mike Lewis 位列第一，`*` 并列贡献标识不改变署名次序。该块还包括其他作者和机构，冻结题目给出的近似上界 155 略早于块底部，但首作者所在行在其内；近似框不是像素级强制分割。

每条实体同时携带论文的 `title` 元数据，但元数据是论文身份信息，不等于任意被引 bbox 内逐字出现的标题。需要证明作者或标题局部内容时，应分别引用上述正文实体或明确、可信且标明来源的元数据，不能用某个后续图注声称含有完整作者列表。

## 补充：Figure 1 多面板与图号未知

BART 第 2 页 Figure 1 原页由 (a) BERT、(b) GPT、(c) BART 三个对照区域组成。解析出了三个图形 crop：`v-96141eccb61ec986bad2ffe1`、`v-639fb535c612d903cb383178` 和 `v-4b77eb65348844428ff5613a`，但三者 `figure_number` 均为 null；它们是单独面板，不是覆盖整幅 Figure 1 的复合图。图内两个模块的 (c) 框不包含 `(c) BART` 说明文字；对应独立正文块 `v-2f00fb22bd1d3c4076391ffd`，整图 Figure 1 图注另在 `v-504b9885e5c0a063225d0a72`。这会影响明确 Figure 1 的约束、跨论文题与图搜图，不能把任何一张单面板或其图注当作整图已完整解析。

除上述目标和 Figure 1 页外，其他图表、未知编号、旋转页面、跨页及全部单元格未做穷尽审核，均保留 unknown。后续任何对 BART 单篇运行的结果也必须标明是单篇局部诊断，不能当成原定两篇语料的无范围首轮评测。
