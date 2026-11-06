# 第二版保留集：Docling 目标证据与原页核查

本核查由未参与选题、实现或模型调用的独立 agent 完成，**不是人工复核或盲审**。仅对照冻结题集、两篇原 PDF、原页渲染与新建 `data/raw/multimodal-holdout-v2-docling/evidence.jsonl` 和 `crops/`；未查看本集检索排名或模型回答，也未改动实现、题集或索引。页数是从 1 开始的 PDF 物理页，bbox 为左上原点 points，以下数值四舍五入，精度以证据文件为准。原页题集核查另见 [执行前复核](MULTIMODAL-HOLDOUT-V2-PRE-AUDIT-20261031.md)。

## 视觉金标：原页区域、图号和图像哈希

这六个目标图表在合并证据中均存在；已对照原页及实际 crop 检查视觉内容，并对每个 crop 独立计算 SHA-256，与该证据的 `image_sha256` 均一致。哈希一致只能证明文件未偏离记录；区域与原文对应是另行目视核对的结果。以下图形框不包含图注，图注单列为正文证据。

| 论文、目标 | 证据 ID、页和 bbox | 面板、内容及 SHA-256 |
|---|---|---|
| N19-1423 Figure 1 | `v-a95e7d5f2b33cf30e5636ebd`，第 3 页 `[75.49,62.83,522.29,239.92]` | 一张完整主图包含预训练和微调两个主要阶段，微调侧有多任务层叠；对应图注另在 `v-bcb662d63ac62b1b64676acc`。SHA `25931937f2370b6ee6aa8907efd4a497ca436b4ed0dac5b0e2e21ad4fc0f5121`。 |
| N19-1423 Figure 2 | `v-aa353350eacefb1f8c95708d`，第 5 页 `[115.93,62.25,471.88,167.77]` | 一张图覆盖输入行和 token、segment、position 嵌入三行及加号；图注另在 `v-2fbf5a8801a0edf308bc1a98`。SHA `6801869e9ff6e18108904a70b7eae77644405a4ae6f9537d169d0f6c7c62172f`。 |
| N19-1423 Table 1 | `v-bd5fb1a3c155e453b94417a8`，第 6 页 `[71.47,63.22,525.41,153.87]` | 一张完整表格包含 BASE 79.6 和 LARGE 82.1 的 Average 列；图注另在 `v-0e9966191a8b45ff97fe190a`。SHA `1b8089c748a8fbd9e6c478f28a543cfab8ae49e63e4acd774fefe0f66d562522`。 |
| P19-1285 Figure 1 | `v-9d13e4571a7ff69ff1dd7cd0`，第 3 页 `[73.37,60.37,522.83,167.39]` | 单张复合图完整包含 (a) Train 和 (b) Evaluation 两个面板及 limited-context 标识；图注 `v-1e4eb1bd3fec5e48315351d4`。SHA `b7956ad11e47fd4a20724b4b693a5dc4f44fe48f461a3565b78d0a1da4fb2b66`。 |
| P19-1285 Figure 2 | `v-2a91c8d9e30bafb52e70bf11`，第 4 页 `[70.47,60.60,517.00,159.15]` | 单张复合图完整包含 (a) training 和 (b) evaluation 两面板，可见 fixed/no-grad、new segment、extended context；图注 `v-43e329fad972aee9f09d857f`。SHA `bc412bfd1dd4f20d2f415df9a8c0b8d4d59d6afb74a7686c55e0e2bec1d2f6c2`。 |
| P19-1285 Table 1 | `v-aaa51bde12f46c1511122763`，第 6 页 `[73.32,62.51,293.28,190.00]` | 一张完整表格包含 Standard 24.0 与 Large 18.3 PPL；图注 `v-760a365514f641643633c9e6`。SHA `156b965179947d307db80e328d10a9c2eb9b7e57e751d3c50b2a66e3de8cd1dd`。 |

上述六条图表源的 PDF SHA 分别与冻结 BERT 原件 `987545ffb087f1ece898142c403a516baeabeb70ce19089397fac6f7db12c3d4` 和 Transformer-XL 原件 `2438bdcddc0db3328a78d108df6438c180f97ef6a12c555eb10a5d6e0e4d734b` 一致，页旋转均为 0。现有记录的图注关联为 `docling_link_unverified`；本表对这六个具体关联对照原页已核查，不推广到其他页面。两张表格的机器提取文本保持 `docling_extracted_unverified`，对本题指定数值已另对照裁图和原表，其他单元格未穷尽复核。

## 逐题目标实体映射

这些 ID 用于审阅解析覆盖及后续候选命中，不预设任何检索排名。题目指定的多个证据需分别匹配，不能用同论文/同页的其他不相关区域代替。

| 案例 | 实际解析出的金标目标 |
|---|---|
| v201 | N19 Figure 2 图 `v-aa353350eacefb1f8c95708d`；题集列为替代证据的图注正文 `v-2fbf5a8801a0edf308bc1a98`，不是额外必需的第二目标。 |
| v202 | N19 Figure 1 图 `v-a95e7d5f2b33cf30e5636ebd` **及**图注正文 `v-bcb662d63ac62b1b64676acc`；共享初始化来自图注，不能用只含图形的框冒充。 |
| v203 | N19 Table 1 图 `v-bd5fb1a3c155e453b94417a8`。 |
| v204 | N19 第 3 页正文 `v-7c7dfc257c4fed7da8703472`，bbox `[307.28,528.21,525.55,618.88]`，同一段含两种模型层数及参数量。 |
| v205 | P19 第 4 页 Figure 2 图 `v-2a91c8d9e30bafb52e70bf11`；第 3 页 3.2 节前文 `v-de9db05380e0050f74462976` 说明前段缓存、固定及扩展上下文，正文 `v-9430f2cd13d4c5067c84ad78` 明确解释 stop-gradient，公式块 `v-f9260889ac59fd83c32cf5e3` 提供 SG 记号。图上 fixed/no-grad 也能看出冻结，但若声称对正文中公式符号的完整解释，须考虑这多个文本块。 |
| v206 | P19 第 3 页 Figure 1 图 `v-9d13e4571a7ff69ff1dd7cd0`；正文 `v-6c0fc2e992a434c5407ea63a` 讨论固定长度语境局限，`v-69faa7d5902f8ccc38b011c3` 描述评估时固定长度窗口按一个位置滑动。前一块开头承接上一页文本，开头单词截断，不影响所列局限段落。 |
| v207 | P19 第 6 页 Table 1 图 `v-aaa51bde12f46c1511122763`。 |
| v208 | 两个必要图：N19 Figure 2 `v-aa353350eacefb1f8c95708d` 和 P19 Figure 2 `v-2a91c8d9e30bafb52e70bf11`。 |
| v209 | N19 Figure 2 `v-aa353350eacefb1f8c95708d`；无范围的唯一性只在固定两篇论文内成立。 |
| v210 | 无唯一目标；两个可指对象为 N19 Figure 1 `v-a95e7d5f2b33cf30e5636ebd` 与 P19 Figure 1 `v-9d13e4571a7ff69ff1dd7cd0`。新会话无前文时澄清，不计入固定目标召回分母。 |
| v211 | 无可核查的文件哈希目标；原 PDF 可搜索文本的独立扫描未见 SHA-256 字符串或 64 位十六进制值，外部文件/附件与不可搜索图像仍 unknown。 |
| v2img01–03 | 依次对应 N19 Figure 2 `v-aa353350eacefb1f8c95708d`、P19 Figure 1 `v-9d13e4571a7ff69ff1dd7cd0`、P19 Figure 2 `v-2a91c8d9e30bafb52e70bf11`。查询图独立由原页生成，不是上述 crop 文件的复制。 |

## 已观察到的多面板关联缺陷

目标六张图表的面板完整不代表通用多面板解析已解决。对 BERT 第 13、15 页另行渲染原页核查发现：

- Figure 3 原页有 BERT、OpenAI GPT、ELMo **三个**并列面板。证据拆为 `v-df9660e8e0430a2ae3f80bec`、`v-03a692599e8364bc1c49c212`、`v-8d60953881342f242879f909` 三个区域，前两者编号 unknown，只有最右 ELMo 面板带 Figure 3 及描述完整三模型的图注；该单面板不能代表完整 Figure 3。
- Figure 4 原页有 (a)–(d) **四个**任务面板。证据拆为 `v-1ed7df96b39a26052eedaa36`、`v-33ab507cd65017ca9a3c44b3`、`v-1fadb2c82fd7f90a79f1ba5c`、`v-469ed84deebd7177c427646e`，只有右下面板带 Figure 4 和完整图注；该单面板不能代表完整 Figure 4。

这两处不属于本次固定目标，依然是真实解析限制；不得因这六个目标完整就断言多面板普遍正确。其他图表、漏图、跨页、所有单元格和非零旋转未逐一核查，均为 unknown。本报告没有检查实际 Top-5、回答、语义自检或请求结果；应由独立的结果审查另行对照，且均还需要人工复核。
