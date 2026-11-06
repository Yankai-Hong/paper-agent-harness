# 第三版论文隔离保留集

固定题集 [cases.json](cases.json) SHA-256 为 `833d3ea10bea410672a2974e3e1de5e1788c859ca34c83adcab950f6a97d6a1d`。两篇原论文、问题、期望答案、范围、近似原页区域及三张图查询均在任何项目 Docling 解析、索引、检索或模型调用前确定。[源码及配置冻结清单](../../../reports/revision/MULTIMODAL-HOLDOUT-V3-FREEZE-20261101.json)在首次解析前写入；随后 ULMFiT 的零宽 bbox 导致解析中止，完整两论文评测未完成，仅对 BART 执行了单论文部分诊断。实际分母、失败和结果见[阶段报告](../../../reports/revision/MULTIMODAL-V4-20261101.md)。不得基于新集输出修改实现或金标后继续称其独立保留集。

| 论文及官方来源 | 原始 PDF SHA-256 | 物理页 |
|---|---|---:|
| [ULMFiT, ACL 2018，P18-1031](https://aclanthology.org/P18-1031/)；[原文](https://aclanthology.org/P18-1031.pdf) | `357805f5bbe6b0672e9cceabce4ccbdcb09943d7e1d567de0cea1f593aca1352` | 12 |
| [BART, ACL 2020，2020.acl-main.703](https://aclanthology.org/2020.acl-main.703/)；[原文](https://aclanthology.org/2020.acl-main.703.pdf) | `3562c723fe7bebb8b362533122b3ca5c065037c8e8d67460654cf3740652bb59` | 10 |

两篇均非此前五篇、ELMo、Sentence-BERT、BERT 或 Transformer-XL；本地 `data/qasper/papers.jsonl` 和先前多模态题集按 ID、完整标题搜索未见匹配。论文引用其他已见方法不代表它们进入本集索引。根据 [ACL copyright FAQ](https://aclanthology.org/faq/copyright/)，ACL 2016 年及以后的材料使用 CC BY 4.0；第三方材料不自动包含。论文原件、原页及查询图保存在本地忽略目录 `data/raw/multimodal-holdout-v3-20261101/`，不随仓库发布。使用者须自行从官方 HTTPS 下载并校验。

## 题目与证据口径

9 条问答中，v301–v305、v308、v309 共 **7 条有确定答案**；v306 是独立会话无前文的真实歧义，v307 的权重文件摘要应为 unknown，不计入目标召回分母。三条图搜图另以 **3** 为分母。目标 Recall@5 / MRR 与完整证据召回应将前置误澄清、请求错误视为任务级未命中，同时另列实际运行检索的分母；不要将扩展证据当成最终 Top-5。v301、v302、v304、v305 各有两个不可互换的必要目标；v308 的标题、作者有可能由独立原页文本块分别承载，不能以论文元数据假装某一局部截图逐字包含它们。

| 案例 | 原页证据与边界 |
|---|---|
| v301 | ULMFiT p3 Figure 1 的 a/b/c 三面板与其独立图注；图注说明三个具体阶段。 |
| v302 | BART p3 Figure 2 的 text infilling，以及同页左栏 Text Infilling 正文；单个 mask 与 Poisson λ=3 应分别可核查。 |
| v303 | BART p5 Table 1 的 XSum PPL 列，两种设置依次为 6.61、6.62；不得把该列解释成其他数据集或 BLEU。 |
| v304 | ULMFiT p6 Table 2 的 IMDb/TREC-6 两项值 4.6、3.6，图注限定为测试错误率 (%)；表框与图注应独立定位。 |
| v305 | 同一句按 BART、ULMFiT 顺序给出两个精确 paper ID；“前者/后者”按本句顺序，证据分别是 BART p2 Figure 1 的 BART 面板与 ULMFiT p3 Figure 1 三面板。 |
| v306 | 新会话中两篇均有 Figure 1；应澄清所指论文，若已有确定会话上下文则不能机械套用该金标。 |
| v307 | ULMFiT 原 PDF 的可搜索文本未发现 `SHA-256` 或 64 位十六进制摘要；不能推断图像、外部权重、附件及其他版本不存在。 |
| v308 | BART p1 完整题名和署名列表；Mike Lewis 是第一署名作者，标题与论文身份可由可信来源元数据支持，但元数据不属于任意所引图注框。 |
| v309 | 无范围时，ULMFiT p3 Figure 1 在这两篇论文内可唯一识别为三个阶段；不对其他论文作唯一性结论。 |

`bbox_points_approx` 为原始 PDF 左上原点 points 的目视近似区域，非像素级金标；必须对照原页判断模型引用的页和 bbox。ULMFiT Figure 1 的三个子图与 BART Figure 1 的三个对照区域均应检查实际裁剪是否完整，不能因单面板附有完整图注即判作完整图。表格文本和图片哈希也不能单独证明语义支持。以上原页由本 agent 检查，尚需不参与题目制作的另一 agent 复核，均不是人工复核或盲审。

## 独立图搜图查询

三张查询图由原 PDF 用 pypdfium2 4.30.0 以 1.5 倍（108 dpi）渲染、按 fixture 中的矩形裁剪并以 Pillow 11.3.0 bicubic 缩放，与项目稍后生成的索引裁图独立。v3img01 为 ULMFiT Figure 1 三面板，v3img02 为 BART Figure 1 三个比较区域及其原图内说明，v3img03 为 BART Figure 2 示意；除 v3img02 的原图内小字外，不含图下独立图注。不得从目标标签预置检索结果。下载原文并校验后，可在新目录复现：

```powershell
& .\.venv-mm-20261030\Scripts\python.exe scripts/prepare_multimodal_holdout_v3.py --pdf-root data/raw/multimodal-holdout-v3-20261101 --output data/raw/multimodal-holdout-v3-reproduced-new
```

本地已在独立输出目录重建三张查询图，三者 SHA-256 均与 fixture 相同；尚未使用项目解析、模型或检索对本集运行。
