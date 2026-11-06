# 多模态真实 PDF 解析修复（开发复验）

本次仅修复与验证解析；不更换模型，也未调用远程模型。V3 的 ULMFiT 和 BART 已参与诊断，以下结果都是**开发复验，不是独立保留集成绩**。此前冻结文件、首次 ULMFiT 失败日志、旧解析、索引、账本与结果保留。本次原页由执行修复的 agent 核对，既非独立盲审，也非人工复核。

## 修复和边界

`scripts/docling_ingest.py` 将每个无法可靠定位的区域分别记录到新建的 `*.quarantine.jsonl`，包含论文、原 PDF 哈希、物理页、Docling 引用、源字符范围、原坐标、页尺寸、原因和文本哈希；没有有效 chunk ID 或可引用裁图。区分零宽、零高、两者均零、非有限数、反向及越界；正常小区域保留，不扩张或更改任何框。缺少来源的条目也记录并标明未知页。无有效文本的条目逐项记为 `skipped/empty_text`，不能当作内容完整；它们与坐标隔离分开计数。PDF 页号越界、Docling 缺页、无效页尺寸、转换非完整成功、全部无可用证据、旋转不受支持等结构性问题仍失败。路径与文件已有检查保持，输出文件、遗漏记录和摘要均拒绝覆盖。

开发复验源码 `scripts/docling_ingest.py` SHA-256：`d4f4af3f8b192f26add5cc6d90db95f39f2f115aea6f73a9b65cbe40b0c678e7`；受控复核脚本 SHA-256：`a0c87a2e5a1f0e86d65de6d9adfa1b55637e95397567979279ebe94c3b48b86c`。此结果不是原冻结版本成功，原失败仍在 `output/mm-holdout-v3-docling-p18.log`。

## 原页抽查和真实解析

- ULMFiT：PDF SHA-256 `357805f5bbe6b0672e9cceabce4ccbdcb09943d7e1d567de0cea1f593aca1352`，12 页、186 项证据、13 个图表候选；10 个多区域源项目拆分，字符范围未知 0。第 2 页 `#/texts/27` 的 `[400.273,592.924,400.273,602.302]` 零宽框单独隔离；同页其他 11 项仍输出。对照原页，相关右栏正文有独立有效区域 `v-2825a099a73fe6d01f94cebd`，但不能据此断言零宽条目的文本已被完整恢复或把它当作合法引用。第 4 页 3 条、第 5 页 1 条 `empty_text` 与原页公式区域对应，公式内容仍缺失，未伪造公式；完整性需要后续检查。
- BART：PDF SHA-256 `3562c723fe7bebb8b362533122b3ca5c065037c8e8d67460654cf3740652bb59`，10 页、180 项证据、15 个图表候选，未隔离；13 个多区域源项目拆分。第 2 页三个 Figure 1 面板仍分别保存在原始输出。原页核对后，受控清单将三个源面板组合为新整体 `v-f545f55690e131b015c8a0ae`，bbox `[68,60,535,343]`，裁图包含 (a)、(b)、(c) 图及其说明；整体 Figure 1 图注 `v-504b9885e5c0a063225d0a72` 保持独立 bbox。三个原面板、坐标和图片哈希保存在 `bart-reviewed/panels.jsonl`，没有修改旧证据。对应图片 SHA-256 `d42a59b46bae200abef6a043a11e12ba9dc022db21b474fe5ab5906274cfbbe3`。
- BART 第 5 页表 `v-7e971512098372bc26f89a4c` 与单独文本 `v-00f66a7a9464f883e5e2489f` 经同一 agent 对照原页确认为 Table 1；Docling 将后者标成普通文本而不是 caption，原自动规则因此保留 unknown。本次只通过明确清单、核对 PDF 哈希、两个源 ID、页和精确 bbox 建立 `agent_page_reviewed_not_human` 关联；表图和图注仍可分别定位。该受控机制不是自动根据邻近猜号，也不证明其他表格已正确关联。

原始逐页统计如下：`证据/图表/未知编号/未知图注关系/隔离/空文本跳过`。未知关系数不等于所有错误关系数：Docling 声称的关联仍为未验证。

| 物理页 | ULMFiT | BART |
|---:|---|---|
| 1 | 17/0/0/0/0/0 | 13/0/0/0/0/0 |
| 2 | 11/0/0/0/1/0 | 19/3/3/3/0/0 |
| 3 | 14/1/0/0/0/0 | 23/1/0/0/0/0 |
| 4 | 18/1/0/0/0/3 | 22/2/2/2/0/0 |
| 5 | 16/1/0/0/0/1 | 16/1/1/1/0/0 |
| 6 | 12/2/0/0/0/0 | 19/2/0/0/0/0 |
| 7 | 15/4/2/2/0/0 | 16/5/1/1/0/0 |
| 8 | 14/3/0/0/0/0 | 10/1/1/1/0/0 |
| 9 | 12/1/0/0/0/0 | 20/0/0/0/0/0 |
| 10 | 24/0/0/0/0/0 | 22/0/0/0/0/0 |
| 11 | 26/0/0/0/0/0 | — |
| 12 | 7/0/0/0/0/0 | — |

复核版 BART 共 178 项、13 个图表；第 2 页由三个未知单面板变成一个明确复核的整体，第 5 页 Table 1 有明确复核关联，其余第 4、7、8 页共 4 个候选仍编号及图注关系未知。ULMFiT 第 7 页仍有 2 个候选未知。图像路径、坐标和 SHA-256 分别对原 ULMFiT 的 13 张、原 BART 的 15 张及复核 BART 的 13 张核验；哈希一致不能证明语义或所有区域完整。

逐页机器统计在 `output/mm-parse-repair-v2-{ulmfit,bart,bart-reviewed}-pages.json`；解析、隔离和复核产物在 `data/raw/multimodal-parse-repair-v2-20261102/`；运行日志 `output/mm-parse-repair-v2-20261102-{ulmfit,bart}.log`，本地 trace 在 `runs/multimodal-parse-repair-v2-20261102/`。第一次尝试得到的中间修复版也独立保留在不带 `v2` 的对应目录。

## 环境和复现

Windows，Python 3.10.11，Docling 2.48.0，PyTorch 2.6.0+cu124，pypdf 5.9.0，pypdfium2 4.30.0；请求 CUDA 设备 `cuda:0`，本机 NVIDIA GeForce RTX 4060 Laptop GPU 可用。解析阶段本地 trace 约 ULMFiT 17.80 秒、BART 16.84 秒。权重使用已下载的 `models/docling-mm-20261030-v2`。在受限执行环境中 Python 的 `_ctypes` DLL 初始化失败；正常权限下运行现有独立环境成功，无需修改证书或 `.env`。

以下命令应将输出路径改成**尚不存在的新目录**，不得覆盖本次或旧记录。两次解析共用新的裁剪根，但 paper ID 各不相同：

```powershell
$vpy = '.\.venv-mm-20261030\Scripts\python.exe'
& $vpy scripts/docling_ingest.py data/raw/multimodal-holdout-v3-20261101/P18-1031.pdf --paper-id P18-1031 --image-root data/raw/NEW/crops --output data/raw/NEW/P18-1031.jsonl --device cuda:0 --artifacts-path models/docling-mm-20261030-v2 --trace-dir runs/NEW
& $vpy scripts/docling_ingest.py data/raw/multimodal-holdout-v3-20261101/2020.acl-main.703.pdf --paper-id 2020.acl-main.703 --image-root data/raw/NEW/crops --output data/raw/NEW/2020.acl-main.703.jsonl --device cuda:0 --artifacts-path models/docling-mm-20261030-v2 --trace-dir runs/NEW
$env:PYTHONPATH='src'
& $vpy scripts/apply_reviewed_composites.py data/raw/NEW/2020.acl-main.703.jsonl --overrides reports/revision/MULTIMODAL-PARSE-REPAIR-BART-REVIEW-20261102.json --pdf-directory data/raw/multimodal-holdout-v3-20261101 --source-root data/raw/NEW/crops --output-root data/raw/NEW/reviewed/crops --output data/raw/NEW/reviewed/evidence.jsonl --panel-manifest-output data/raw/NEW/reviewed/panels.jsonl
```

统计脚本 `scripts/summarize_docling_repair.py` 需提供 `--pdf --evidence --image-root --omissions --output`，拒绝覆盖摘要并检查 PDF 与图像哈希。受控清单仅适用于其精确匹配的已复核 PDF、来源实体和坐标；任意论文仍需单独核查。

## 本地定向检索

用复核后的 BART 单论文证据和原固定 CLIP 快照，在 `indexes/multimodal-parse-repair-bart-20261102/` 新建 13 张图的本地索引，512 维、L2 归一化。按论文和 `Figure 1` 过滤的文本查询返回复核整体图第 1；按 `Table 1` 过滤的文本查询返回复核表图第 1。这两个过滤后都只有一个合格候选，说明过滤与关联可用，不能据此声称开放范围语义排名通过。此前的 Figure 1 查询图在无图号过滤的 BART 单篇图搜图中返回完整复合图第 1；用 ULMFiT 论文范围查询此 BART 单篇索引返回 0 项。逐条真实结果保存在 `output/mm-parse-repair-v2-{figure1,table1,image,negative-scope}-search.json`；这不是独立新集或完整问答验收。

## 测试与限制

原完整 GPU 测试环境 `.venv-gpu`：**393 passed、1 skipped、73 warnings**；最终定向解析、路径安全与图表关联测试 **35 passed**；`git diff --check` 通过。精简 `.venv` 的全量结果另存：**388 passed、4 skipped、2 failed**，两项为可选 Langfuse SDK 子进程缺少 `pydantic`，并非远程上传；完整环境复跑通过。仅使用本地模拟 SDK 和本地索引，没有向遥测服务上传论文或图片。本轮真实模型请求及 token 均为 0，历史账本不改。

零宽阻塞已修复但不等于论文完整解析：隔离文本、四个公式、未知编号及其他页的图表归属仍需人工复核。引用格式或图片哈希不是证据语义正确性的证明。
