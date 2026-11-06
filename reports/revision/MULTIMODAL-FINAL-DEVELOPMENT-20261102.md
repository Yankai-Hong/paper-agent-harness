# 多模态第一阶段：双论文开发复验与验收

本次完成有限范围修复和一次真实复验。BART 与 ULMFiT、题目和原页此前已用于诊断与修复，以下全部是**开发复验**，不是独立保留集、盲测或论文库泛化准确率。原页及结果由本轮执行者 agent 核查，不是人类金标。原始 PDF、失败日志、冻结记录、旧索引、旧结果和旧账本均保留。

**验收判断：真实 PDF → 图表 → 索引和检索 → 模型看图／回答 → 本地引用核查的基础链路在本双论文范围内跑通；第一阶段完整验收仍未通过。** 标题／作者题未完成，部分回答或引用缺少限定和独立图注，历史 h07 过度拒答未解决，解析仍有隔离和遗漏，不能据此宣称泛化或语义全面正确。

## 本轮修改及边界

- 在新目录合并两篇来源证据，分别建立文本、CLIP 与图注／表格语义索引。BART Figure 1 整体和 Table 1 关联来自 [原页 agent 复核清单](MULTIMODAL-PARSE-REPAIR-BART-REVIEW-20261102.json)，不是一般多面板自动识别；来源面板和独立图注仍保留。ULMFiT 第 2 页零宽条目仍隔离，第 4–5 页四个公式仍未提取，其他未知图表关联见 [解析报告](MULTIMODAL-PARSE-REPAIR-20261102.md)。
- 历史 v303 的 trace 仅显示当时 `subqueries=18/18`，没有保存异常具体资源项，**确切触发项仍 unknown**。本次为多模态两次检索单独将子查询上限设为 32，其余限制仍为 240 秒、8 次模型请求、320 候选和 180 次重排；新的 `BudgetExceeded` 本地 trace 和案例结果会记录白名单资源项，不泄露任意异常内容。v303 本次未再发生异常，但不能倒推出旧异常原因。
- 已有同句明确前者／后者规则及真实歧义守卫通过测试，本次 v305 按顺序回答、v306 保守澄清；此前 v205 的原始 NLU 不在账本中，具体 schema 原因依然 unknown。
- 增加首页原文候选：仅当题目同时要求标题和作者、明确单论文范围、存在唯一与标题元数据逐字相符的有效首页文本和其后可定位文本时才加入候选；作者仍由证据判断，来源元数据保持 `unverified`，不当作框内正文。**当前入口范围识别仅接受严格冒号前缀**，v308 普通的“论文 ID”提问未触发，仍拒答；不能宣称该问题已修复。
- h07 中 `this effort` 的引用段可由原页消解，但现有指代保护仍会过度拒答；本轮未引入可能将歧义指代自动判为支持的启发式，也未复跑 h07，仍是未修复项。数值与 test 条件检查、引用核查和模型自检仍启用；模型自检并非独立语义证明。

## 九问逐例结果

Top-5 指实际融合排序前五；“扩展证据”指 Top-5 以外由管线加入、仍可单独引用的来源。目标逐个排名与所有通道候选见本地 `runs/multimodal-final-dev-20261102/retrieval-review.json`，实际生成、所有声明和引用在 `predictions.json` 与 `claim-audit/`；引用坐标均为 PDF 物理页、左上原点 points。表中语义判断为本轮 agent 对原页的判断，不把自检结果当成人类验证。

| 案例 | 实际检索与决定 | 原页、回答和引用判断 |
|---|---|---|
| v301 | ULMFiT 图第 1；独立图注未在 Top-5，作为扩展证据加入；answer | `stage_a/b/c` 分别引用 p3 图 `v-048bf54e3c62918316858d40` 与图注 `v-634d07fc63ec7383ca4980cb`。图中三阶段名称及图注对应，引用支持已说的名称；答案未明确展开通用领域及目标任务限定，按题目预期记**部分完整**。 |
| v302 | Text Infilling 正文第 1、Figure 2 第 2；answer | `span_replacement`、`poisson_lambda` 均引用 BART p3 正文 `v-6eb2eb027e9afb995f1da312`，单个 `[MASK]` 和 λ=3 均获该区域直接支持，答案正确。图已检索但最终未引用；数值不是从图框推断。 |
| v303 | Table 1 表图第 2；answer，无预算异常 | 两条分别给出 XSum PPL 6.61、6.62，表中行列与数字正确；两条只引用 p5 表图 `v-7e971512098372bc26f89a4c`。**表号在框外独立图注** `v-00f66a7a9464f883e5e2489f`，虽该图注位列 Top-5 第 5，却未被引用；数字支持、完整表号引用条件部分缺失。 |
| v304 | Table 2 表图第 2，独立图注不在 Top-5但经扩展加入；answer | `imdb_ulmfit_error`、`trec6_ulmfit_error` 引 ULMFiT p6 表图 `v-f023738b85a3e064777c0d81` 与图注 `v-959951e0c42d4a564d285ed4`；4.6、3.6 的行列由图支持，test error rates 与 % 由独立图注支持。回答和引用支持。 |
| v305 | BART 复核整体图第 2，ULMFiT 整图第 4；answer | `bart_figure1_modules` 引 BART p2 整图 `v-f545f55690e131b015c8a0ae`，包含 (c) BART 说明及两个模块；`ulmfit_figure1_stages` 引 ULMFiT p3 图 `v-048bf54e3c62918316858d40`，`ulmfit_stages_caption` 引独立图注 `v-634d07fc63ec7383ca4980cb`。同句顺序、模块及三阶段限定均由相应区域支持。 |
| v306 | 未检索，clarify | 新会话中两篇都有 Figure 1，要求明确论文符合题意；无最终声明、引用和适用的目标命中分母。 |
| v307 | 检索后 refuse，`visual_or_text_evidence_insufficient` | 未伪造摘要；原 PDF 的可搜索文字未见 SHA-256，不能据此证明外部文件或不可搜索区域不存在摘要。无最终声明和适用的指定目标分母。 |
| v308 | 首页标题、作者块均未进 Top-5；refuse | 原 PDF p1 标题 `v-8b5eae33d2488c0053126e62` 与署名 `v-1ddb443c2b36c90376d92bc3` 能回答，首位为 Mike Lewis；本次没有最终引用和答案，**任务失败**，不能用索引标题替代原文。 |
| v309 | ULMFiT Figure 1 第 4；answer | `paper_figure1_three_stages` 引 ULMFiT p3 正文 `v-5e770711c4c21a9ddde86d17`、独立图注 `v-634d07fc63ec7383ca4980cb` 和图 `v-048bf54e3c62918316858d40`；三阶段及论文身份受支持，唯一性只适用于此双论文语料。 |

七条有指定可回答目标的问题中，至少一个指定目标进入 Top-5 为 **6/7**，所有预先列出的目标进入 Top-5 为 **4/7**；v301、v304 的图注虽不在 Top-5，但被实际扩展并引用。v308 两个目标均未入 Top-5。不能把“有目标命中”当作完整回答或语义正确。9 问中 6 条生成答案、1 条明确澄清、2 条拒答；无异常或未执行。v301 不完整、v303 表号来源未独立引用、v308 未完成，故不能把 6 个 answer 全部算作完整通过。v306/v307 为行为题，不加入上述 7 题检索目标分母。

上述主要被引 bbox：ULMFiT p3 图 `[71.865,62.303,531.086,236.462]`、图注 `[72,249.493,525.546,313.055]`；BART p3 正文 `[72,224.645,290.271,353.137]`；BART p5 表 `[79.106,62.626,517.701,263.220]`，其独立图注 `[72,280.066,525.547,348.394]` 未被 v303 引用；ULMFiT p6 表 `[99.640,62.511,497.768,142.916]`、图注 `[95.5,155.054,502.049,164.419]`；BART p2 复合图 `[68,60,535,343]`。精确值、各声明和图片哈希以 `predictions.json` 为准。模型自检对六个有答案的案例均判各声明 supported；本 agent 不因此忽略上面所列缺失限定和图注。

## 三条无范围图搜图

以下列表是双论文索引实际无论文、无图号过滤的 Top-5，均按顺序列实体短 ID（前缀均为 `v-`）；不是 oracle 结果：

| 案例 | Top-5 | 完整目标 |
|---|---|---|
| v3img01 | `048bf54e3c62918316858d40`, `ca9e7933ca38ace3a8f0324c`, `800f7400f57b2fd1afdedac0`, `9a647d18795af46569c671f3`, `1b6c8e1c5e3355e1e1183488` | ULMFiT p3 Figure 1 完整图第 1 |
| v3img02 | `f545f55690e131b015c8a0ae`, `ca9e7933ca38ace3a8f0324c`, `32418e972cc7b006c6005474`, `a4e9cf505dc170fd9903bd70`, `048bf54e3c62918316858d40` | BART p2 Figure 1 的 agent 复核完整复合图第 1 |
| v3img03 | `a4e9cf505dc170fd9903bd70`, `ca9e7933ca38ace3a8f0324c`, `32418e972cc7b006c6005474`, `f545f55690e131b015c8a0ae`, `048bf54e3c62918316858d40` | BART p3 Figure 2 完整图第 1 |

完整目标 **3/3** 进入无范围 Top-5，均排第 1；不是任意复杂图的泛化结论。另单独以真实论文范围、图号及两者共同过滤运行 oracle 检查，三条目标均第 1，所有返回项满足相应过滤，joint 过滤各仅 1 项；这些 oracle 排名不与无范围成绩混合。各项完整 paper ID、物理页、bbox 和三个 oracle 列表保存在 `predictions.json`。

## 可复现记录、测试与调用

新语料和裁图：`data/raw/multimodal-final-dev-20261102/`；文本、CLIP、语义索引分别为 `indexes/multimodal-final-dev-{text,clip,semantic}-20261102/`。合并后 338 文本块、26 视觉项；模型与先前一致，CLIP 512 维、文本 BGE 1024 维。原始证据来自解析修复 v2 和 BART 受控复核版，原始隔离清单未进入索引。各索引的 corpus/model/index 哈希在其 `metadata.json` 及本轮 `config.json`；源码和题集哈希在本轮 `source-hashes.json`。

实际调用 `scripts/run_multimodal_v3_eval.py`，参数为上述三索引、`--visual-root data/raw/multimodal-final-dev-20261102/crops --visual-python .\.venv-mm-20261030\Scripts\python.exe --visual-image-model models/multimodal-v1/clip-image-v1 --visual-text-model models/multimodal-v1/clip-text-v1 --cases data/eval/multimodal-holdout-v3/cases.json --fixture-sha256 833d3ea10bea410672a2974e3e1de5e1788c859ca34c83adcab950f6a97d6a1d --decompose --encoder models/bge-m3 --reranker models/bge-reranker-v2-m3 --device cpu --visual-device cuda:0`；输出使用全新 `runs/multimodal-final-dev-20261102/`。本地模型离线加载，遥测明确关闭。重新运行须使用新输出和索引路径并独立记账，不能覆盖本次记录。

本轮独立账本 **30 次真实 provider 请求，30 成功、0 失败、30 次 usage 完整**：输入 119,486、输出 4,006，共 **123,492 token**；请求累计耗时约 111.03 秒，实际费用 unknown。每次请求、状态、provider 原始 usage 和耗时在 `runs/multimodal-final-dev-20261102/requests/`；图搜图没有远程请求。旧 60 次预算标记和历史账本均只读保留，本轮依用户新授权另记。测试：完整 `.venv-gpu` **395 passed、1 skipped、73 warnings**，定向范围、引用、首页及预算投影 **83 passed**，`git diff --check` 通过。未上传 PDF、图片或凭据到 Langfuse。

这次只验证两篇 PDF 和原 12 个案例。零宽隔离项、四个公式、其他未知图表关联、跨声明指代误拒以及首页范围入口仍需后续处理；更大论文集、通用公式识别和任意复杂多面板自动关联未验证。本轮到此停止，不因小样本结果开启新题集或继续自动调参。
