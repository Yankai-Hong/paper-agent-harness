# Chroma 迁移与 IRPAPERS 运行记录

本记录区分已完成的真实运行与未验证部分。此前四个本地提交、原索引、原始数据和评测记录均未重置；本任务不推送。旧的完整 FAISS IRPAPERS 基线未启动，相关脚本与数据保留，后续基线改用 Chroma。

## Chroma 检索迁移

默认新建的 BGE 正文、BGE 图注／表格表示、CLIP 图像和 CLIP 文本索引使用 Chroma 0.6.3；BM25 仍由权威文本快照构建。每个索引使用独立持久化路径，和已有情景记忆存储分离；元数据保留语料、模型、维度及归一化签名。Chroma 使用 cosine distance，适配器转换为 `1 - distance` 的余弦相似度，原 0.90/0.05 图片门槛因此仍作用在相似度上。Chroma 的 HNSW 排序为近似检索，不能将其当作 FAISS IndexFlat 的精确保证。

从已校验的旧 FAISS 向量及来源映射分别复制到新目录，不修改旧索引：

| 用途 | 新目录 | 数量 | 维度 | 逐向量最大绝对差 | 五个查询的 Top-5 集合一致 |
|---|---|---:|---:|---:|---:|
| BGE 正文 | `indexes/chroma-migration-text-20261106` | 338 | 1024 | 1.50e-8 | 5/5 |
| BGE 图注／表格表示 | `indexes/chroma-migration-semantic-20261106` | 38 | 1024 | 0 | 5/5 |
| CLIP 图表 | `indexes/chroma-migration-clip-20261106` | 26 | 512 | 0 | 5/5 |
| CLIP 文本 | `indexes/chroma-migration-clip-text-20261106` | 374 | 512 | 5.97e-8 | 5/5 |

比较使用相同向量，不是模型效果评测；真实 BGE 的两条查询 Top-5 排序也一致。迁移后的 CLIP 开发检索实际运行，图搜图 3/3，图注限定直接检索 2/3，文搜图融合 1/2，与先前小样本结果相同。详细比较文件位于 `output/chroma-migration-*-verify-20261106.json`。完整回归 **434 passed、1 skipped**，包含情景记忆测试；模型、论文图片和旧环境未迁移或删除。FAISS 只作为可选旧索引读取与迁移对照依赖，安装时可选择 `.[legacy-faiss]`。

向量 collection 构建期支持重复 upsert、删除和重启续建；封存后只读，变更须创建新版本，避免向量与来源快照分离。封存记录 ID 与向量指纹；中断留下未封存目录并拒绝作为完成索引读取。复杂坐标及原文保持在单独的来源快照，不能从向量推断 citation。

## IRPAPERS 数据核验

数据来源为 [IRPAPERS 数据集](https://huggingface.co/datasets/weaviate/IRPAPERS)，公开版本 `7d8ca2f6dd9efded3e27013d15782d584f93e9da`，数据卡标记 MIT，底层论文版权可能不同。本地 3,777,129,846 字节页面 CSV 的 SHA-256 为 `8fe316290b55c71bd1641603b5feb02117672028f7d1af45bcad1c111c53611b`，与该版本 LFS 哈希一致；两个问题 CSV 的 Git blob 哈希也与该版本一致。逐页解码结果：3,230 行、166 篇论文、3,230 张有效图片、4 页转写为空，未发现重复页面 ID 或图片解码失败。完整统计在 `data/raw/irpapers-prepared-20261106/audit.json`；本地审计脚本不输出或记录 Base64。

180 个标准问题中一条的 `dataset_id=16_7` 与 `pdf_id=15,page_number=7` 冲突，不能自动判定真实目标；另有一条目标 `12_9` 转写为空。冲突按未知保留在 180 的分母中，并单独记录两种标签的排名；空文本目标也不静默排除。30 个视觉问题与标准问题有 13 篇论文和 19 个目标页重叠，但没有完全相同的页面＋问题；它们不是独立论文保留集。[原论文](https://arxiv.org/abs/2602.17687)说明页面转写由 GPT-4.1 生成；这不是本项目 Docling 解析或坐标引用评测。

已按论文关联分组、随机种子 20261106 冻结划分，冲突涉及的论文属于同一组：标准问题开发 138、保留 42；视觉问题开发 19、保留 11。最终基线计划和代码／模型哈希在 `runs/irpapers-chroma-baseline-v2-20261106/plan.json`，SHA-256 `68f532337951eb3acd3333c1b48dc805da2e905ce29e1befd89bae3496ada975`。3,226 个非空页面按 BGE tokenizer 的 500 token、80 token 重叠规则生成 9,166 个片段；完整语料不按目标论文、页码或答案过滤。

## 实际全语料基线

首次运行的 BM25 210 条成功，但 210 条稠密查询和 210 条混合查询因 Chroma 0.6.3 对该索引的全量 HNSW 请求报 `Cannot return the results in a contigious 2D array` 而失败，原结果保留于 `runs/irpapers-chroma-baseline-20261106/`，不能记为零召回。适配器对这一特定错误读取 Chroma 中的向量并执行精确余弦全扫描，记录 `chroma_exact_full_scan_after_hnsw_limit`；较小查询仍使用近似 HNSW。v2 是新的索引与冻结计划，9,166 条向量与原编码索引逐项核对，最大差 `2.98e-8`。v2 的 210 条文本和 210 条页面视觉查询均成功，全量排名均使用上述明确标记的精确回退；不将其当作 HNSW 吞吐量或 FAISS 检索。

下面数值为 Hit@1 / Hit@5 / Hit@20 / MRR。原始逐题排名和汇总分别在 `runs/irpapers-chroma-baseline-v2-20261106/{text-results.jsonl,visual-results.jsonl}` 及其 `.summary.json`。冲突案例按未知计零且保留分母，相关总体数字为保守口径；空转写页面无法进入文本索引，也保留分母。

| 集合 | 方法 | Hit@1 | Hit@5 | Hit@20 | MRR |
|---|---|---:|---:|---:|---:|
| 标准全部 180 | BM25 | .4500 | .7778 | .8944 | .5911 |
| 标准全部 180 | BGE | .4000 | .7333 | .8611 | .5487 |
| 标准全部 180 | 文本混合 | .3944 | .7722 | .9111 | .5606 |
| 标准开发 138 | 文本混合 | .3696 | .7754 | .8986 | .5443 |
| 标准测试 42 | BM25 | .4524 | .8810 | .9524 | .6294 |
| 标准测试 42 | BGE | .4048 | .7143 | .8333 | .5560 |
| 标准测试 42 | 文本混合 | .4762 | .7619 | .9524 | .6142 |
| 视觉全部 30 | BM25 | .1333 | .4000 | .5000 | .2286 |
| 视觉全部 30 | BGE | .2000 | .3333 | .5000 | .2810 |
| 视觉全部 30 | 文本混合 | .1667 | .4000 | .6000 | .2794 |
| 视觉开发 19 | 文本混合 | .1053 | .3158 | .4737 | .2122 |
| 视觉测试 11 | 文本混合 | .2727 | .5455 | .8182 | .3954 |

| 集合 | 页面 CLIP 或页面等权融合 | Hit@1 | Hit@5 | Hit@20 | MRR |
|---|---|---:|---:|---:|---:|
| 标准全部 180 | CLIP | .0056 | .0278 | .0667 | .0187 |
| 标准全部 180 | 等权融合 | .0833 | .3000 | .8222 | .2075 |
| 标准开发 138 | CLIP | .0072 | .0362 | .0870 | .0237 |
| 标准开发 138 | 等权融合 | .0942 | .3043 | .8333 | .2165 |
| 标准测试 42 | CLIP | .0000 | .0000 | .0000 | .0023 |
| 标准测试 42 | 等权融合 | .0476 | .2857 | .7857 | .1779 |
| 视觉全部 30 | CLIP | .0000 | .0000 | .0333 | .0095 |
| 视觉全部 30 | 等权融合 | .1000 | .2333 | .5000 | .1915 |
| 视觉开发 19 | CLIP | .0000 | .0000 | .0526 | .0137 |
| 视觉开发 19 | 等权融合 | .1579 | .2632 | .4211 | .2099 |
| 视觉测试 11 | CLIP | .0000 | .0000 | .0000 | .0022 |
| 视觉测试 11 | 等权融合 | .0000 | .1818 | .6364 | .1598 |

该 CLIP 索引编码整页图片而非图表裁剪，固定图像与多语言文本快照、归一化及预处理写在视觉汇总中；结果不衡量细粒度图表定位、回答或引用。开发问题中没有中文题，不能据此评价中文。基线在优化前冻结：`runs/irpapers-chroma-baseline-v2-20261106/BASELINE-FROZEN.json` SHA-256 `697b917a1e852ffc3845cf90b5d338ca1a2242a9f7486648c91cbbca0f313d13`。

BGE 全语料编码约 318.25 秒，峰值显存 1,171,011,584 字节；页面 CLIP 的 3,230 页 GPU 编码约 991.25 秒，峰值显存 641,319,936 字节，随后 Chroma 写入约 4.5 秒。分别是不同进程的实测，不能相加解释为单次查询延迟。完整视觉索引约 12.5 MB；分片与本地检查点未计入此数。该实验未调用远程生成模型，新增请求和 token 均为 0。

## 仅使用开发集选择融合

在 138 条标准和 19 条视觉开发问题上，按冻结规则比较文本片段级与页面级 RRF，页面级虽然 MRR 较高，但 Hit@5 下降，故保留片段级。比较 CLIP 权重 0、0.05、0.1、0.25、0.5、1 后，选择权重 **0**：这实际退回文本混合，不能声称 CLIP 带来增益。选择及候选值在 `runs/irpapers-chroma-optimization-20261106/plan.json`，SHA-256 `7c4791fbb94d1de51b5629056f05a787012b0796fb9bfebd01f6ef22f91c0fc1`；选择时未查看测试题逐例结果。

冻结选择后，标准测试 Hit@5 从等权融合 12/42 升至 32/42，视觉测试从 2/11 升至 6/11；逐题分别 20 和 4 个改善，均无退步。两者与原有文本混合**完全相同**，因此只是避免等权 CLIP 的损害，不是超越文本基线。完整结果和逐例在 `runs/irpapers-chroma-optimization-20261106/results.jsonl` 及汇总；融合结果冻结文件为 `OPTIMIZATION-FROZEN.json`。冻结计划和结果均未覆盖首轮失败记录。

## 复现与边界

在项目根目录、已准备数据与两个独立环境的前提下，可用新的输出名运行以下步骤。冻结脚本绑定源码和模型哈希，必须使用与计划一致的源码；每次选取全新路径，现有模型、索引和运行不得覆盖。实际设备和模型见计划，不要在复现时假设 GPU 可用。

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'
$vpy = '.\.venv-mm-20261030\Scripts\python.exe'
$src = 'data/raw/irpapers'
$prepared = 'data/raw/irpapers-prepared-NEW'
$plan = 'runs/irpapers-NEW/plan.json'
$chunks = 'runs/irpapers-NEW/chunks.jsonl'
& $py scripts/irpapers_prepare.py --source $src --output $prepared
New-Item -ItemType Directory -Path 'runs/irpapers-NEW'
& $py scripts/irpapers_baseline.py freeze --source $src --prepared $prepared --output $plan
& $py scripts/irpapers_baseline.py chunks --plan $plan --prepared $prepared --output $chunks
& $py scripts/irpapers_baseline.py text-build --plan $plan --chunks $chunks --output indexes/irpapers-text-NEW
& $py scripts/irpapers_baseline.py text-eval --plan $plan --source $src --index indexes/irpapers-text-NEW --output runs/irpapers-NEW/text-results.jsonl
& $py scripts/irpapers_baseline.py visual-build --plan $plan --prepared $prepared --visual-python $vpy --request runs/irpapers-NEW/visual-request.json --checkpoint data/raw/irpapers-clip-NEW --output indexes/irpapers-page-clip-NEW
& $py scripts/irpapers_baseline.py visual-eval --plan $plan --source $src --text-results runs/irpapers-NEW/text-results.jsonl --visual-index indexes/irpapers-page-clip-NEW --visual-python $vpy --output runs/irpapers-NEW/visual-results.jsonl
```

本轮不启动 ColQwen2。五篇论文的真实 Docling/CLIP 与 10 问、3 图搜图结果独立记录于 [第一阶段报告](MULTIMODAL-2026-10-31.md) 和 [复核报告](MULTIMODAL-AGENT-REVIEW-20261031.md)：助手及独立 agent 的核查均不是独立人工金标。第一阶段持久预算已用满 60 次，不得通过新账本绕过。

当前源码的完整回归为 **437 passed、1 skipped、238 warnings**（54.39 秒），包含原有文本和情景记忆测试。受限环境的首次尝试为 430 passed、1 skipped、7 failed，失败均涉及本机 HTTP 测试且报告连接 `PermissionError`；允许本地连接后上述完整回归通过。`git diff --check` 通过。离线回归不作为真实视觉语义通过的证据。
