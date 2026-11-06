# 图片到文本与图片问答：双论文开发评测

本轮实现真实的图片到文本检索及用户图片参与问答，独立于先前第一阶段报告。**链路能执行，但图片到目标图注的直接召回与纯图片回答相关性不足，不能宣布检索质量全面通过。** 两篇论文、题目和三张图均已用于此前开发；以下是小样本开发复验，不是独立保留集、盲审或人类金标。目标在未加论文、图号或答案范围限制的完整索引中实际存在，仅用于事后评分。

## 实现和索引

- CLIP 文本索引与 BGE 正文及 CLIP 图像索引分别存储。图像快照 `sentence-transformers/clip-ViT-B-32` commit `327ab6726d33c0e22f920c83f2ff9e4bd38ca37f`；文本快照 `sentence-transformers/clip-ViT-B-32-multilingual-v1` commit `58edf8cada9e398793dca955574a48cbb7f18be2`。模型快照文件哈希分别 `85d6c4d8881ccf203ce6b9c9f664fca849ec175e84ec939592baa35fb9c418e3`、`3739808321eb3cedc449e4c732dea9088c55c83762b6f0888077be761a603d20`。原 BGE 分块未改写，不计算跨模型向量相似度。
- 本机 GPU 编码 `indexes/multimodal-clip-text-v2-20261106/`：2 篇论文、338 个有页面坐标的文本源（含 25 个图注）、374 个表示、512 维；图像索引 26 个实体。每个表示有原 chunk、字符偏移、论文和位置，fast tokenizer 最大 128 token，减去 2 个特殊 token、126 token 窗口、重叠 16。源、表示、模型和 FAISS 哈希及 count 在 metadata 中；图像与文本索引的模型和 PDF 哈希也在问答入口核对。表格结构或 OCR 未经独立定位核查时不升级为已验证原文。
- `PaperQuery` 携带可选文本和至多两张本地用户图片；状态通过原会话对象传入。旧 `run(question, state)` 保持可用。一次图片编码由图片→文本与图片→图片分支共享；独立正文和图像按各自排名交错、按原 ID 去重，不把不同实体算作共同命中。候选图注仅在同论文、同 PDF、同页、文字唯一匹配等保护下作为独立来源扩展，关系仍标未核实。用户图没有论文引用身份；引用必须回填真实论文文本或图表的位置。
- 图片限 PNG/JPEG、20 MiB、4000 万像素，模型每次请求最多三张图，用户图占用其中的名额。真实远程模型请求会发送用户图及选中的论文图；本地 trace 和请求账本不记录原图或 base64。索引缺失或模型、数据、维度不匹配明确报错；图片两分支的 ok、empty、failed 状态保存在 trace。带图查询中旧文本和普通视觉分支可在局部失败时降级；可选 BGE 图注分支的错误仍可能终止整轮，应视为限制。

## 真实检索

完整结果、Top-5、来源和索引元数据在 `output/clip-image-text-eval-v3-20261106.json`。每项仅有一个指定目标，所以此处 Hit@5 与逐目标 Recall@5 数值相同；关联扩展单独列示，不作为直接 CLIP 命中。

| 路径 | 问题/图片数 | 指定目标 Hit@5、Recall@5 | 说明 |
|---|---:|---:|---|
| 文搜正文 BM25 | 1 | 1/1 | BART Text Infilling 正文第 1。 |
| 文搜图 CLIP | 2 | 0/2 | 英文 ULMFiT 阶段和中文 BART 填空查询均未召回指定图；中文例第 1 是 ULMFiT Figure 1。 |
| 图搜图 CLIP | 3 | 3/3 | 三个原定目标均第 1；同为结构图的另一论文作为困难负例进入候选。 |
| 直接图搜正文 CLIP | 3 | 0/3 | 指定的三个独立图注均不在 Top-5；返回的确为原始文本，有些结果与查询无关。 |
| 图搜图后扩展独立图注 | 3 | 3/3 | 每次从检索到的图按来源关联找到指定图注；不是直接文本索引命中，扩展也包括其他图的图注。 |

图搜图的 BART Figure 1 是先前由 agent 对照原页制作的受控复合图，不能推广为一般多面板自动解析。真正长文本诊断使用索引中第二个表示的原始文本作为查询，目标 `P18-1031/v-f0357b9308136b25e5bda3da` 的字符区间 513–688 排第 1；这是自查询诊断，不能当作独立相关性评测。

## 真实问答与引用

最终源码混合与纯图片运行在 `runs/clip-image-question-smoke-final-v2-20261106/`，无标准答案前缀或预先指定论文范围，使用已有 OpenCode Go `deepseek-v4.1-flash`，1024 输出 token、60 秒、无自动重试。Reflect 和 Generate 均实际收到用户图和论文图。混合问题询问图中阶段，目标图在融合 Top-5 排第 3，目标图注不在融合 Top-5，但作为独立来源扩展；目标图与图注两项的融合 Top-5 Recall 为 1/2，全部找回为否，扩展后均进入证据。最终四条声明给出三个阶段及通用领域、目标任务限定，分别引用 ULMFiT p3 图 `v-048bf54e3c62918316858d40` 和独立图注 `v-634d07fc63ec7383ca4980cb`。助手复看了原 PDF 第 3 页，图、图注位置和这些结论一致；这是助手核查，不是人类复核。

纯图片回答找到了同一图、图注和正文，相关三阶段内容受原页支持，但还加入无关的 BART 分类图及 ULMFiT Figure 2，**相关性部分失败**。更早一轮纯图片返回澄清，另一轮拒答，还有一轮包含无关声明；这些变动及所有账本保留，不能只挑成功案例宣称稳定。前一次混合复验曾因未识别用户图片而澄清；当前代码对受限的图片指代放宽范围，同时保留明确论文、历史指代和不确定范围保护。单例模型输出与合法引用格式不证明通用语义正确。

## 请求、测试及复现

本轮六个本地账本合计 22 次请求尝试，20 次 provider 返回成功、2 次受限运行环境报 `PermissionError` 且 provider usage unknown；成功请求的 provider 汇报输入 77,969、输出 5,579、总计 83,548 token，费用 unknown。两个受限尝试未收到响应，不能将未知用量写成已确认的零。多个账本包括过渡和失败，不可与此前阶段请求混淆。最终完整回归 **424 passed、1 skipped、73 warnings**，包括纯文本回归、旧视觉测试和本轮新增的模式、来源、分片、维度及失败分支测试；`git diff --check` 通过。该测试集含替身单元测试，真实评测单独如上。

在项目根目录使用下列命令复现检索；输出文件必须不存在。模型、数据和索引准备及正常 CLI 问答命令见 `docs/multimodal.md`。真实问答脚本额外使用 `--output runs/新目录` 并产生远程调用费用。

```powershell
$env:TEMP = (Resolve-Path .cache).Path
$env:TMP = $env:TEMP
& .\.venv-gpu\Scripts\python.exe scripts/eval_clip_image_text.py --clip-text-index indexes/multimodal-clip-text-v2-20261106 --visual-index indexes/multimodal-acceptance-repair-clip-20261102 --visual-root data/raw/multimodal-final-dev-20261102/crops --visual-python .venv-mm-20261030/Scripts/python.exe --image-model models/multimodal-v1/clip-image-v1 --text-model models/multimodal-v1/clip-text-v1 --device cuda:0 --output output/clip-image-text-eval-new.json
& .\.venv-gpu\Scripts\python.exe -m pytest -q --tb=short
```

剩余限制：直接图搜正文与文搜图召回低；纯图问答容易引入无关候选；多图、多领域和更大语料未实测。图注关联、表格 OCR/提取、图中细小数字和箭头仍需独立核查，CLIP 分数与同模型自检均不证明语义；上述题目和目标均尚未独立人工复核。此前 Docling 几何隔离、漏公式、v305/v309 等限制继续保留。
