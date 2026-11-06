# 第二版论文隔离保留集：执行前独立原页复核

复核者为未参与题目制作、实现和模型调用的独立 agent；原页和金标**尚未独立人工复核**，也不是严格盲审。本核查只看冻结题集、原始 PDF、原页渲染和查询图；未查看或运行本集 Docling、项目检索或模型输出。PDF 页码为从 1 开始的物理页，题集区域是左上原点 PDF points 的**近似框**，不是像素级标注。未修改冻结题集、原 PDF 或实现。

## 来源、版本与复现

ACL Anthology 的 [BERT 原条目](https://aclanthology.org/N19-1423/) 和 [Transformer-XL 原条目](https://aclanthology.org/P19-1285/) 分别确认论文题名、ID、2019 年及对应 PDF；[ACL 版权说明](https://aclanthology.org/faq/copyright/)确认 2016 年及以后的 ACL 材料适用 CC BY 4.0，第三方材料不自动包含。不能由此推断每项可能的第三方材料授权状态。

独立计算的 fixture SHA-256 为 `662f643fb38e95299c230ae7ae640ab0a6c5e33bad7817d9feb2b765c1443de6`。BERT 原 PDF 16 页、SHA-256 `987545ffb087f1ece898142c403a516baeabeb70ce19089397fac6f7db12c3d4`；Transformer-XL 原 PDF 11 页、SHA-256 `2438bdcddc0db3328a78d108df6438c180f97ef6a12c555eb10a5d6e0e4d734b`，均与题集一致。分别在原 PDF 对目标页以 1.5 倍渲染，按冻结矩形裁剪和双三次缩放后，与三份查询文件逐像素一致且文件 SHA-256 均匹配。上述证明可重现，并不证明后续系统的任何解析或检索结果正确。

## 逐例原页核查

| 案例 | 核查结果 | 原页依据与界限 |
|---|---|---|
| v201 | 一致 | BERT 物理第 5 页 Figure 2 的三行分别是 token、segment、position embeddings，图中加号与图注支持相加；所列图及独立图注近似区域对应。 |
| v202 | 一致 | BERT 第 3 页 Figure 1 标出预训练与微调；图注说明不同下游任务从相同预训练模型参数初始化。图注位于图形下方，应作为独立定位证据，不能只凭图形 bbox 认定已包含图注。 |
| v203 | 一致 | BERT 第 6 页 Table 1 的 Average 列，BASE 和 LARGE 分别为 79.6、82.1；该表注释明确 Average 与官方 GLUE 分数略有区别。 |
| v204 | 一致 | BERT 第 3 页右栏 Model Architecture 给出 BASE 12 层/110M，LARGE 24 层/340M；近似框覆盖该段，注意不要把 110M/340M 误写为精确参数计数。 |
| v205 | 一致 | Transformer-XL 第 4 页 Figure 2 标示固定、无梯度的缓存和新片段；第 3 页 3.2 节说明前段隐藏状态固定缓存为下一段扩展上下文并定义 stop-gradient。两页均为所需证据。 |
| v206 | 一致 | Transformer-XL 第 3 页 Figure 1 分为 train 和 evaluation；图中 evaluation 的有限上下文与正文对固定长度及滑动评估的说明相符。不能解读为论文声称只训练或评估一次。 |
| v207 | 一致 | Transformer-XL 第 6 页 Table 1 是 WikiText-103，Standard 24.0 PPL、Large 18.3 PPL；对应左上表格近似框。 |
| v208 | 一致 | BERT 第 5 页 Figure 2 为三种嵌入相加；Transformer-XL 第 4 页 Figure 2 为跨段状态复用、固定缓存及扩展上下文。两张图均需核查，不可只用同号图混同论文。 |
| v209 | 一致，有语料边界 | 在冻结的两篇论文中，题述 Figure 2 对应 BERT 第 5 页；唯一性仅在该两篇语料内成立，不是全球唯一声明。 |
| v210 | 一致，有会话前提 | BERT 与 Transformer-XL 均在物理第 3 页有不同 Figure 1；独立新会话无论文指代时需要澄清。如真实会话含足够先前上下文，不能直接沿用该金标。 |
| v211 | unknown 处理合理 | 独立扫描 BERT 全 16 页可搜索文本，未发现 `SHA-256` / `SHA256` 或连续 64 位十六进制值；这不排除不可搜索图像、外部权重、附件或其他版本出现摘要，不能断言外部文件不存在。 |

## 查询图核查

| 案例 | 原页及目标 | 复现与裁剪 |
|---|---|---|
| v2img01 | BERT 第 5 页 Figure 2 | 渲染、裁剪、缩放后尺寸 440×134；文件 SHA-256 `887e070f17db9aacbebb62f7adf77b43b5508d2e1b747f12236d07c9e1a1ae0d`。主体、三行标签和加号可见；未见实质截断。 |
| v2img02 | Transformer-XL 第 3 页 Figure 1 | 尺寸 549×132；SHA-256 `617a2d5dd15d77ac7761c66e8cc8556c4a0082ac7aa873b1cd6543cde6dc3ed2`。训练和评估两面板、对应标签可见；未见实质截断。 |
| v2img03 | Transformer-XL 第 4 页 Figure 2 | 尺寸 617×138；SHA-256 `6ae0c2d90193be0381e96457f788ae87c9fa8e0856911adeb1c08dd0e828a6cc`。训练、评估面板以及 fixed/no-grad 与扩展上下文标签可见；未见实质截断。 |

三图的边缘留白较少、并非像素精确的完整图形分割，且查询图均不含下方图注；实际评测时不得把查询裁剪当作图注证据。其他未检查页面的解析完整性、旋转及跨页情况均为 unknown。固定题集不保证目标必然被 Docling 索引。
