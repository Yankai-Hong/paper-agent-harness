# 第二版论文隔离保留集

本集 11 问和 3 条图搜图的题目与金标已在本集 Docling 解析、项目检索、模型调用之前固定；`cases.json` 的 SHA-256 是 `662f643fb38e95299c230ae7ae640ab0a6c5e33bad7817d9feb2b765c1443de6`。当时另存了[实现和配置冻结清单](../../../reports/revision/MULTIMODAL-HOLDOUT-V2-FREEZE-20261031.json)，随后已执行首次验证，结果见[阶段报告](../../../reports/revision/MULTIMODAL-V3-20261031.md)。如依据本集结果改变实现、阈值或题目，须将其降为诊断集，再另建保留集。

两篇公开 ACL 论文均不属于原来的五篇开发论文（1903.10318、1909.06708、1909.12079、1911.04474、2002.06854），也不是已经用于诊断的 ELMo N18-1202 或 Sentence-BERT D19-1410。在本地 `data/qasper/papers.jsonl` 中按完整标题和论文 ID 搜索未发现这两篇。两篇为新的隔离语料；论文中引用旧方法并不代表旧论文进入这个索引。名称、来源、页数和原 PDF SHA-256 固定如下：

| 论文 | 原始版本 | SHA-256 |
|---|---|---|
| N19-1423 | [BERT，NAACL 2019](https://aclanthology.org/N19-1423/)，[PDF](https://aclanthology.org/N19-1423.pdf) | `987545ffb087f1ece898142c403a516baeabeb70ce19089397fac6f7db12c3d4` |
| P19-1285 | [Transformer-XL，ACL 2019](https://aclanthology.org/P19-1285/)，[PDF](https://aclanthology.org/P19-1285.pdf) | `2438bdcddc0db3328a78d108df6438c180f97ef6a12c555eb10a5d6e0e4d734b` |

[ACL 版权说明](https://aclanthology.org/faq/copyright/)对 2016 年以后 ACL 材料适用 CC BY 4.0；其中第三方材料不自动包含在许可内。原 PDF、页面图像和查询裁剪位于忽略目录 `data/raw/multimodal-holdout-v2-20261031/`，不提交仓库；重现者须自行从上述官方 HTTPS 地址取得并校验哈希。按用户授权，仅向现有 OpenCode Go 模型发送测试所需的公开论文上下文与图表，不上传其他服务或 Langfuse。

## 核查与计数

一个 agent 对照两篇**原 PDF 页面**制作并核查题目、答案、页码和区域；这不是独立人工复核，也不是模型结果的盲评。执行前建议由未参与制作的另一 agent 独立复核原页及金标，记录分歧，不按系统输出修改金标。页码为 PDF 物理页（从 1 开始）；`bbox_points_approx` 是视觉估算的 PDF points 左上坐标，**不是像素级边界真值**。必须实际对照原页核查被检索出的各个区域；坐标或来源不能确定时标为 unknown，不能只凭页面或图号断言正确。

- v201–v207 是明确单论文范围，中英题均有；v202、v205、v206 要求或显式结合图与图注/正文。v201 图注明确作为替代证据，不当作必须同时出现的第二个金标。
- v208 是明确两篇论文比较；v209 是无范围但可在**本两篇语料**唯一识别的目标；v210 无对话历史且两篇均有 Figure 1，应澄清；v211 检索无可核查的文件哈希时应保留 unknown，不声称证明外部文件不存在。
- v2img01–03 为原页不同区域生成的查询图，与解析产生的图像实体不是同一文件；后两条是易混淆的 Transformer-XL Figure 1/2。无范围运行后可另外报告论文和图号过滤结果，不能混同。
- 有确定目标的可回答问题为 9 题（v201–v209）；v210/v211 不属于目标召回分母。问题级至少一个金标 Recall@5 和 MRR 需将提前误澄清计为未命中；完整证据召回逐个核对 `evidence` 所列全部必要证据，不把候选池扩大当作最终 Top-5。位置正确、语义支持和事实正确分别评估；每条声明、图注与图形单独核查。三条以图搜图为另一独立分母。

## 生成和复现查询裁剪

固定页面用 `pypdfium2 4.30.0` 在 `scale=1.5`（108 dpi）渲染，Pillow 11.3.0 按 `cases.json` 所列像素矩形裁剪并用 bicubic 缩放；源 PDF 和查询裁剪的哈希均固定。下列命令先验证两份 PDF，输出目录必须不存在，并验证生成的三张查询图哈希：

```powershell
& .\.venv-mm-20261030\Scripts\python.exe scripts/prepare_multimodal_holdout_v2.py --pdf-root data/raw/multimodal-holdout-v2-20261031 --output data/raw/multimodal-holdout-v2-reproduced
```

原图各自不包含论文其他正文或图注，且没有任何项目索引候选预置为检索结果。图表的可用性、裁剪完整性和图注识别须以真实 Docling 运行及原页复核为准，不能因题集包含某个目标就假定目标已被成功索引。
