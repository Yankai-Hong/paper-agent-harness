# IRPAPERS 纯 ColQwen2 全库评测（2026-11-10）

本记录与[先前候选内评测](COLQWEN-CONTINUATION-20261108.md)分开。目标是对每个文字问题在 3,230 张原始页面上计算真实 MaxSim；不使用 BGE、BM25、CLIP 或 gold 筛选页面，也没有进行多向量压缩。本次语料的页面转写是 benchmark transcription，不是 Docling PDF 原文；图像结果仅为页面级定位，不能作为图表 bbox 或问答引用验收。

## 数据与真实向量

本轮重新计算原始页面、标准题和视觉题 CSV 的 SHA-256，分别为 `8fe316290b55c71bd1641603b5feb02117672028f7d1af45bcad1c111c53611b`、`a87d8263d8b498cdaceb6cb719abcd0a584ded3993910aad41edab1db8fa6ed3` 和 `972aa9f2f1541defbefc4226667dba04653af8be9c8ad5351c201fdc82b89b93`；页面清单 SHA-256 `d127db27db1077edf0f19defc7a643b3071c33cfe56acd0502b40d3a748faac9`。旧审计为 3,230 页、166 篇、标准 180 题和视觉 30 题；视觉题与标准题重叠 13 篇论文、19 个目标页，不能称独立测试集。标准题 `standard-143` 的页面标识冲突保留为 unknown 并计入分母；`standard-090` 目标页转写为空，仍保留。历史保留集汇总已经曝光，不是严格盲测。

开发和保留页面编码契约的模型 revision、两片基础权重、适配器哈希、128 维、256 视觉 token 上限、处理器版本、设备和源页面清单一致。1,361 个重叠页面逐一比较向量（绝对容差 `1e-6`）、图像哈希、网格、有效 token 位置和来源，均通过；2,322 页复用已验证模型向量，其余 908 页用同一模型与隔离环境真实编码。当前运行的处理器配置与先前记录的哈希一致；历史检查点没有绑定所有配置文件哈希，不能追溯性地声称已验证历史每个配置文件。

三张不依赖 gold 选择的真实页面和一个真实开发问题上，GPU 批量 MaxSim、独立逐页计算与官方 processor 的最大差 `1.0431e-6`，原始输出 `output/colqwen-full-probe-20261110.json`。向量、有效 token 和页面来源保留；不将图像 token 当作像素框。

## 进度与恢复

3,230 页向量已齐备在 `runs/colqwen-full-20261110`；新独立 Chroma 索引 `indexes/colqwen-full-20261110` 正逐页导入和校验。**封存及完整评测未完成前，本文件不提供全库指标。** 每页写入记录、输入参考和来源单独保存；中断时以相同命令续建，不覆盖原索引或检查点。若复制中断留下无 marker 的不完整向量文件，先保留并改名该新运行目录中的孤儿文件，再以同一契约续跑；不要删除原数据或旧检查点。

导入启动后为加速后续重启和回读，工作区中的导入脚本增加了批量封存实现；已运行进程继续使用启动时的原实现。启动版本按本次记录的代码改动反向重建并保存在 `runs/colqwen-full-20261110/importer-start-reconstructed.py`，SHA-256 `a3809b1671713db550fa54bc50b61f4fe1849bf76025925584c974af27a38d61`；启动时没有预先保存该文件或哈希，因此这是重建版本，不冒充当时的预先冻结快照。封存后需用当前代码在新进程重新回读核验。

```powershell
$p = '.\.venv-gpu\Scripts\python.exe'
$c = '.\.venv-colqwen-20261106\Scripts\python.exe'
$env:TEMP = (Resolve-Path '.cache').Path
$env:TMP = $env:TEMP
& $c scripts/colqwen_full_encode.py --prepared data/raw/irpapers-prepared-20261106 --dev runs/colqwen-dev-20261107 --test runs/colqwen-test-20261108 --dev-index indexes/colqwen-dev-20261107 --test-index indexes/colqwen-test-20261108 --base models/colqwen2-base-range-20261107 --adapter models/colqwen2-v1-20261106 --output runs/colqwen-full-20261110 --max-seconds 1800
& $p scripts/colqwen_full_import.py --prepared data/raw/irpapers-prepared-20261106 --encoded runs/colqwen-full-20261110 --dev-index indexes/colqwen-dev-20261107 --test-index indexes/colqwen-test-20261108 --adapter models/colqwen2-v1-20261106 --index indexes/colqwen-full-20261110
& $p scripts/colqwen_full_eval.py --prepared data/raw/irpapers-prepared-20261106 --encoded runs/colqwen-full-20261110 --index indexes/colqwen-full-20261110 --baseline runs/irpapers-chroma-baseline-v2-20261106 --dev runs/colqwen-dev-20261107 --test runs/colqwen-test-20261108 --output runs/colqwen-full-eval-20261110 --partition dev
```

评测脚本先冻结源码／索引／模型契约，从 Chroma 批量读取并逐 token 核对元数据与原始向量参考，建立无损页面缓存；GPU 分批计算真实 MaxSim，完整页排名、原始分数与页面来源逐题检查点保存。开发集结果完成后，`scripts/colqwen_full_analyze.py choose` 只使用开发集选择完整页面排名 RRF 权重；保留集计分必须提供冻结选择文件，最后用 `evaluate` 输出比较和逐题退化。暂不将过去 100 页候选重排的 62.22% 称作全库成绩。
