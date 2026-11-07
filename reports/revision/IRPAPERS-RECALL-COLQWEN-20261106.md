# IRPAPERS 召回诊断和 ColQwen2 进展

**结果：标准问题的文本页面融合 Hit@5 提升，但视觉问题退化；尚无真实 ColQwen2 向量或 MaxSim 评测。** 旧索引、冻结基线、数据、未提交修改和历史结果均保留。本轮不更改默认检索，也不推送。未调用远程生成模型，新增 provider 请求和 token 均为 0。

## 整页 CLIP 为什么退化

- 模型快照的本地模型卡明确说明多语言文本编码器与 `clip-ViT-B-32` 图片编码器对齐。快照 revision 与文件哈希见既有[基线报告](CHROMA-IRPAPERS-20261106.md)，运行时校验模型来源。
- 3,230 条预处理页面、编码检查点及 Chroma 索引的页面 ID、论文、页码、图片路径和图片哈希逐行一致；所有 Chroma 向量与编码检查点最大绝对差 `5.96e-8`，结果 `output/irpapers-chroma-page-vectors-verify-20261106.json`。对八张开发样本图重新加载原图并在 GPU 编码，向量与检查点最大绝对差不超过 `5.73e-6`，余弦约 1；原图哈希均匹配。
- 对四个开发问题重新编码文本并对检查点全部 3,230 个页面计算余弦排序，Top-5 与原结果完全一致，目标排名分别重现为 1543、39、1420、170。数据见 `output/irpapers-clip-diagnosis-20261106.json`。这排除了这些样本的反向距离排序和明显的编码、映射错位，不能证明所有数据标签正确。
- 页面原图示例为 2550×3300 或 2481×3508；图像处理器先将短边缩至 224，再中心裁剪 224×224。原始 ViT-B/32 的图像网格仅 7×7，长边还被裁掉一部分。论文小字、表格和局部图形因此难以保留，这是由预处理与尺寸直接观察到的限制；它不能单独证明每条漏召的唯一原因。
- `standard-006` 的问题包含论文专有名称和三阶段过程：文本目标第 1，整页 CLIP 第 1543，等权融合第 14。`visual-008` 涉及嵌入空间位置变化：文本第 3，整页 CLIP 第 1420，融合第 16。它们说明全排名等权 RRF 会让弱视觉分支把可靠文本结果压低；属于开发样本分析，不是人工标签或一般性能证明。
- 所有候选按页面身份合并，全文候选没有按目标页或答案筛选。全量 Chroma 查询触发已记录的应用层精确余弦回退；上述质量结论不是 HNSW 效率结论，HNSW 大 k 问题未解决。
- 将 CLIP 排名截在前 5、20、50、100 页，再试非零权重 0.05、0.1、0.25、0.5、1；要求标准题及视觉题开发集 Hit@5 都不低于文本基线、综合 MRR 提升，没有合格方案。计划及全部开发结果在 `runs/irpapers-windowed-20261106/plan.json`（SHA-256 `16144b517dd3fe96640aee3dfa0bae43d2bb2c359ebac8705a3de86eb540060a`）；因此没有将此分支部署，也没有把权重 0 称为视觉增益。

## 页面级文本融合的实际对照

先将 BM25 和 BGE 各自片段排名按 `page_id` 去重，再对页面做加权 RRF，避免同一页面重复投票。开发集比较 k=10、30、60、120 与 BGE 权重 0、0.25、0.5、0.75、1、1.5、2、4，要求标准题、视觉题的 Hit@5 均不低于已有文本混合，再选综合 MRR 最高者；冻结结果为 k=10、BM25 权重 1、BGE 权重 0.5。计划 `runs/irpapers-page-fusion-20261106/plan.json` SHA-256 `1f7e10bacabd02542c47d72f259e08bdbaa2af06e99b8e39bfb211f2dfde2fd6`，按选择后生成 `results.jsonl` 和 `.summary.json`。历史基线保留集汇总先前已知，不是严格盲测。

| 集合／方法 | Hit@1 | Hit@5 | Hit@20 | MRR |
|---|---:|---:|---:|---:|
| 标准开发 138，旧文本 | .3696 | .7754 | .8986 | .5443 |
| 标准开发 138，新页面融合 | .4565 | .7754 | .9130 | .5907 |
| 标准保留 42，旧文本 | .4762 | .7619 | .9524 | .6142 |
| 标准保留 42，新页面融合 | .4286 | .8571 | .9524 | .6073 |
| 视觉开发 19，旧文本 | .1053 | .3158 | .4737 | .2122 |
| 视觉开发 19，新页面融合 | .1053 | .3158 | .4211 | .1986 |
| 视觉保留 11，旧文本 | .2727 | .5455 | .8182 | .3954 |
| 视觉保留 11，新页面融合 | .2727 | .4545 | .7273 | .3739 |

标准保留题 Hit@5 有 4 条改善、0 条退化；Hit@1 为 1 条改善、3 条退化。视觉保留题 Hit@5 为 0 条改善、1 条退化，Hit@20 另有 1 条退化。完整 ID 和逐题排名在汇总与结果中。该方案只作为独立可调用的 `paper_agent.page_fusion.fuse_page_rankings` 和实验脚本保留，不更改现有问答、图片检索、引用或记忆默认行为。它是文本重排，不是视觉增益；标准保留集 BM25 单独的 Hit@5 为 .8810、MRR .6294，优于本方案。

仅离线排名融合的中位耗时约 3.0 ms、最大 23.6 ms；不包含模型编码、Chroma 检索及数据加载。原基线 GPU 显存、RAM 和索引体积见基线报告，不能把离线耗时作为端到端延迟。

## ColQwen2：已验证部分和阻塞

指定 `vidore/colqwen2-v1.0` revision `2b6ac8fb37f46a49e4841e599583d00ae8a20117` 的适配器已下载，权重 SHA-256 `fc856312174dc99a4c7f88a2c54d9590a3b3f5b5a86e2728d7138c7f4758c6d5` 与官方元数据一致；它依赖 `vidore/colqwen2-base` revision `9fe8a713422a7cb4ef79ca77a09b381ee2243101`，两个基础权重分片约 8.84 GB。普通 HTTPS 下载长时间未写入数据；随后使用官方 Xet 组件，首次默认缓存因权限拒绝，改为工作区缓存后两个分片仍长时间为 0 字节，已停止并保留不完整文件。未关闭 TLS 验证，未重复扩大包源。基础权重缺失的实际失败记录在 `runs/colqwen-smoke-missing-base-20261106/result.json`。

新环境 `.venv-colqwen-20261106` 安装了 `colpali-engine 0.3.12`、`transformers 4.53.3`、`peft 0.16.0`、`tokenizers 0.21.4`、`qwen-vl-utils 0.0.11`；实际继承 PyTorch `2.11.0+cu128`，导入成功，但 `pip check` 明确报告 engine 要求 `torch>=2.5,<2.8`，因此环境**尚未符合依赖约束**。从官方 PyTorch CUDA 12.8 源尝试安装 2.7.1 / torchvision 0.22.1 时，约 3.27 GB 的 wheel 也长时间停留在 0 字节，已停止，未安装成功。不得把导入成功当作模型兼容性验证。

本机当时 RTX 4060 Laptop 8,188 MiB 显存、约 6,990 MiB 空闲，系统 16.78 GB RAM、约 4.47 GB 可用。真实开发页 `2_1`（2481×3508）与真实问题已用官方 processor 处理：限制 256 个视觉 token 后输入序列长度 258、问题长度 33、使用 `Query: ` 前缀；`output/colqwen-preprocess-20261106.json` 标明**没有执行模型推理**。显存峰值、模型前向延迟、MaxSim 和最终索引体积均 unknown。

已实现独立的 `paper_agent.colqwen_store.ColQwenPageStore`：Chroma 保存逐页逐 token 向量及图片哈希，封存后校验内容，按候选页读取，并对每个查询 token 求页面 token 最大点积后求和。测试只用合成向量验证运算、页映射、去重和防篡改；不是模型结果。真实编码器到 Chroma 的批量导入、全候选评估与端到端集成尚未完成。`scripts/colqwen_smoke.py` 准备在基础权重完整并校验 SHA-256 后，以已有文本候选逐页执行真实小样本模型前向，核对其 MaxSim 与官方 processor；它未成功运行，也不自动将 smoke 当成基准。

依据开发数据冻结候选：文本混合前 50 页和整页 CLIP 前 50 页并集，按页面去重，每题最多 100 页，不使用答案、目标页或论文范围过滤。候选计划 SHA-256 `dbea772bc28840b5f236839ec88a46c4c64b667b92bf164b7878e29c60e3f32d`，测试在冻结后计算但不属严格盲测：

| 集合 | 候选覆盖 | 平均候选数 | 不同页面数 |
|---|---:|---:|---:|
| 标准开发 | 131/138 | 97.20 | 2043 |
| 标准保留 | 40/42 | 98.81 | 1378 |
| 视觉开发 | 14/19 | 98.26 | 806 |
| 视觉保留 | 9/11 | 98.36 | 608 |

冲突标签 `standard-143` 按原口径未知计入分母；`standard-090` 原页转写为空，仍计入分母。候选结果 `runs/irpapers-colqwen-candidates-20261106/results.jsonl` 只是覆盖率，不是 ColQwen2 召回率或 MaxSim 排序，也不能推断页面的图表区域或引用正确。

## 恢复与复现

下列实验命令使用已冻结的 v2 基线和本机数据；`NEW` 必须换成不存在的新路径。选择步骤只读取开发题，评估步骤在选择文件写入之后才读取全部题；先前已知保留集汇总，因此结果仍不能称为严格盲测。

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'
$vpy = '.\.venv-mm-20261030\Scripts\python.exe'
$baseline = 'runs/irpapers-chroma-baseline-v2-20261106'
& $vpy scripts/irpapers_clip_diagnose.py --device cuda:0 --output output/irpapers-clip-diagnosis-NEW.json
& $py scripts/irpapers_verify_page_vectors.py --index indexes/irpapers-chroma-page-clip-baseline-20261106 --checkpoint data/raw/irpapers-page-clip-checkpoint-20261106 --output output/irpapers-page-verify-NEW.json
& $py scripts/irpapers_windowed_fusion.py choose --baseline $baseline --output runs/irpapers-window-NEW/plan.json
& $py scripts/irpapers_page_fusion.py choose --baseline $baseline --output runs/irpapers-page-NEW/plan.json
& $py scripts/irpapers_page_fusion.py evaluate --baseline $baseline --plan runs/irpapers-page-NEW/plan.json --output runs/irpapers-page-NEW/results.jsonl
& $py scripts/irpapers_colqwen_candidates.py choose --baseline $baseline --output runs/irpapers-candidates-NEW/plan.json
& $py scripts/irpapers_colqwen_candidates.py evaluate --baseline $baseline --plan runs/irpapers-candidates-NEW/plan.json --output runs/irpapers-candidates-NEW/results.jsonl
```

请在能稳定访问 Hugging Face 文件存储和 PyTorch 官方 wheel 的环境，从项目根目录运行。以下命令不覆盖原索引或已有环境；下载使用现有未完成目录续传，日志和 Xet 缓存限定在工作区。安装足够的磁盘空间并保留 TLS 验证。

```powershell
$py = '.\.venv-colqwen-20261106\Scripts\python.exe'
$env:HF_HOME = (Join-Path (Get-Location) '.cache/hf-colqwen')
$env:HF_XET_CACHE = (Join-Path (Get-Location) '.cache/hf-colqwen/xet')
& $py -m pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
& $py -m pip check
& $py -c "from huggingface_hub import snapshot_download; snapshot_download('vidore/colqwen2-base', revision='9fe8a713422a7cb4ef79ca77a09b381ee2243101', local_dir='models/colqwen2-base-20261106', allow_patterns=['*.json','*.safetensors','*.txt','*.tiktoken'], max_workers=2)"
& $py scripts/colqwen_smoke.py --base models/colqwen2-base-20261106 --adapter models/colqwen2-v1-20261106 --output runs/colqwen-smoke-NEW --device cuda:0 --max-visual-tokens 256
```

成功后仍须核对真实模型适配器加载、编码张量、显存与耗时，实现可恢复的候选页面向量导入并在冻结候选内运行 MaxSim，再按原分组逐题评估；OOM 时可在新运行目录尝试较低视觉 token 上限或 CPU，并记录设置和实际资源。候选集合外页面不会被重排，不能声称全库 ColQwen2 检索。使用 [官方模型卡](https://huggingface.co/vidore/colqwen2-v1.0) 与 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)核对依赖。

本轮完整回归 **442 passed、1 skipped、262 warnings**，包括原文本、图片分支和情景记忆测试；单测中的合成向量不构成真实模型验收。`git diff --check` 单独核查。
