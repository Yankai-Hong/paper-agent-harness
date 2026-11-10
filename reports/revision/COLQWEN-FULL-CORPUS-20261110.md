# IRPAPERS 纯 ColQwen2 全库评测（2026-11-10）

本记录与[先前候选内评测](COLQWEN-CONTINUATION-20261108.md)分开。目标是对每个文字问题在 3,230 张原始页面上计算真实 MaxSim；不使用 BGE、BM25、CLIP 或 gold 筛选页面，也没有进行多向量压缩。本次语料的页面转写是 benchmark transcription，不是 Docling PDF 原文；图像结果仅为页面级定位，不能作为图表 bbox 或问答引用验收。

## 数据与真实向量

本轮重新计算原始页面、标准题和视觉题 CSV 的 SHA-256，分别为 `8fe316290b55c71bd1641603b5feb02117672028f7d1af45bcad1c111c53611b`、`a87d8263d8b498cdaceb6cb719abcd0a584ded3993910aad41edab1db8fa6ed3` 和 `972aa9f2f1541defbefc4226667dba04653af8be9c8ad5351c201fdc82b89b93`；页面清单 SHA-256 `d127db27db1077edf0f19defc7a643b3071c33cfe56acd0502b40d3a748faac9`。旧审计为 3,230 页、166 篇、标准 180 题和视觉 30 题；视觉题与标准题重叠 13 篇论文、19 个目标页，不能称独立测试集。标准题 `standard-143` 的页面标识冲突保留为 unknown 并计入分母；`standard-090` 目标页转写为空，仍保留。历史保留集汇总已经曝光，不是严格盲测。

开发和保留页面编码契约的模型 revision、两片基础权重、适配器哈希、128 维、256 视觉 token 上限、处理器版本、设备和源页面清单一致。1,361 个重叠页面逐一比较向量（绝对容差 `1e-6`）、图像哈希、网格、有效 token 位置和来源，均通过；2,322 页复用已验证模型向量，其余 908 页用同一模型与隔离环境真实编码。当前运行的处理器配置与先前记录的哈希一致；历史检查点没有绑定所有配置文件哈希，不能追溯性地声称已验证历史每个配置文件。

页面编码检查点沿用项目已有 `.npz` 无损文件压缩，解码后保留所有 float32 token；这不是降维、池化或近似检索。Chroma 保存各 token，计分不使用 Chroma 的近似向量搜索。

本轮逐个校验 210 条查询检查点的题文哈希、向量文件哈希、维度、有限值与归一化；有效 token 数 27–66。209 个标签一致问题的目标页全部在 3,230 页来源清单中；冲突题的两个可能页面也都存在，但目标仍为 unknown。该覆盖核查不参与在线候选筛选，也不是检索命中结果。

三张不依赖 gold 选择的真实页面和一个真实开发问题上，GPU 批量 MaxSim、独立逐页计算与官方 processor 的最大差 `1.0431e-6`，原始输出 `output/colqwen-full-probe-20261110.json`。向量、有效 token 和页面来源保留；不将图像 token 当作像素框。

再从两套旧候选索引均未覆盖的 908 个新编码页面中按来源顺序取前三页，与同一真实问题进行相同数值对照，三种实现最大差 `1.9074e-6`；页面选择不依赖 gold。结果 `output/colqwen-full-new-pages-probe-20261110.json`。两次探针均只验证数值实现，不是召回率。

## 进度与恢复

3,230 页向量已齐备在 `runs/colqwen-full-20261110`；新独立 Chroma 索引 `indexes/colqwen-full-v2-20261110` 正分批导入和校验。**封存及完整评测未完成前，本文件不提供全库指标。** 每页写入记录、输入参考和来源单独保存；中断时以相同命令续建，不覆盖原索引或检查点。若复制中断留下无 marker 的不完整向量文件，先保留并改名该新运行目录中的孤儿文件，再以同一契约续跑；不要删除原数据或旧检查点。

导入启动后为加速后续重启和回读，工作区中的导入脚本增加了批量封存实现；已运行进程继续使用启动时的原实现。启动版本按本次记录的代码改动反向重建并保存在 `runs/colqwen-full-20261110/importer-start-reconstructed.py`，SHA-256 `a3809b1671713db550fa54bc50b61f4fe1849bf76025925584c974af27a38d61`；启动时没有预先保存该文件或哈希，因此这是重建版本，不冒充当时的预先冻结快照。封存后需用当前代码在新进程重新回读核验。

为启用批量封存，在 1,799 页已提交、1,800 页有写入意图时中断原进程；两次尝试恢复都在未完成页面附近退出且没有 Python 堆栈，第二次明确返回 Windows 原生访问异常 `0xc0000005`，第一次仅记录到退出码 1，原因尚未确定。原索引 `indexes/colqwen-full-20261110`、待完成页面和诊断日志均保留，不能称其已恢复或封存。新建 v2 目录从已验证向量重新导入；批量写入的中断续写、来源篡改和封存守卫已用真实 Chroma 的合成向量回归，这不是模型效果结果。

v2 启动后立即保存了其源码快照：导入脚本 SHA-256 `c683b8ae0b44ba1c3bab5449f286d452fe5a5827419b3c3566270a62eb77020b`，批量存储实现 SHA-256 `2b25dccfab1fa51064ca5ee2cece8f515c10bed74886eea24d8a28efa7795388`；快照和日志位于全库运行目录。另在既有封存索引上真实批量读取 16 页、4,178 个 token，与逐页读取最大差为 0；该检查不衡量检索效果。908 页新增编码的首末检查点时间间隔约 342 秒，不含首个编码前的模型加载；全轮显存峰值未独立采样，不能将历史烟测峰值直接当成本轮实测。

```powershell
$p = '.\.venv-gpu\Scripts\python.exe'
$c = '.\.venv-colqwen-20261106\Scripts\python.exe'
$env:TEMP = (Resolve-Path '.cache').Path
$env:TMP = $env:TEMP
& $c scripts/colqwen_full_contract_check.py --dev runs/colqwen-dev-20261107 --test runs/colqwen-test-20261108 --prepared data/raw/irpapers-prepared-20261106 --base models/colqwen2-base-range-20261107 --adapter models/colqwen2-v1-20261106 --config-hashes reports/revision/colqwen-config-hashes-20261108.json --output output/colqwen-full-contract-NEW.json
if ($LASTEXITCODE -ne 0) { throw '模型或数据契约不匹配，停止运行' }
& $c scripts/colqwen_full_encode.py --prepared data/raw/irpapers-prepared-20261106 --dev runs/colqwen-dev-20261107 --test runs/colqwen-test-20261108 --dev-index indexes/colqwen-dev-20261107 --test-index indexes/colqwen-test-20261108 --base models/colqwen2-base-range-20261107 --adapter models/colqwen2-v1-20261106 --output runs/colqwen-full-20261110 --max-seconds 1800
& $p scripts/colqwen_full_import.py --prepared data/raw/irpapers-prepared-20261106 --encoded runs/colqwen-full-20261110 --dev-index indexes/colqwen-dev-20261107 --test-index indexes/colqwen-test-20261108 --adapter models/colqwen2-v1-20261106 --index indexes/colqwen-full-v2-20261110
& $p scripts/colqwen_full_eval.py --prepared data/raw/irpapers-prepared-20261106 --encoded runs/colqwen-full-20261110 --index indexes/colqwen-full-v2-20261110 --baseline runs/irpapers-chroma-baseline-v2-20261106 --dev runs/colqwen-dev-20261107 --test runs/colqwen-test-20261108 --output runs/colqwen-full-eval-20261110 --partition dev
$common = @('--baseline','runs/irpapers-chroma-baseline-v2-20261106','--full','runs/colqwen-full-eval-20261110','--candidates','runs/irpapers-colqwen-candidates-20261106/results.jsonl','--candidate-dev','runs/colqwen-dev-20261107/results.jsonl','--candidate-test','runs/colqwen-test-20261108/results.jsonl')
& $p scripts/colqwen_full_analyze.py choose @common --output runs/colqwen-full-selection-20261110.json
& $p scripts/colqwen_full_eval.py --prepared data/raw/irpapers-prepared-20261106 --encoded runs/colqwen-full-20261110 --index indexes/colqwen-full-v2-20261110 --baseline runs/irpapers-chroma-baseline-v2-20261106 --dev runs/colqwen-dev-20261107 --test runs/colqwen-test-20261108 --output runs/colqwen-full-eval-20261110 --partition test --selection runs/colqwen-full-selection-20261110.json
& $p scripts/colqwen_full_analyze.py evaluate @common --plan runs/colqwen-full-selection-20261110.json --output runs/colqwen-full-comparison-20261110.json
```

评测脚本先冻结源码／索引／模型契约，从 Chroma 批量读取并逐 token 核对元数据与原始向量参考，建立无损页面缓存；GPU 分批计算真实 MaxSim，完整页排名、原始分数与页面来源逐题检查点保存。开发集结果完成后，`scripts/colqwen_full_analyze.py choose` 只使用开发集选择完整页面排名 RRF 权重；保留集计分必须提供冻结选择文件，最后用 `evaluate` 输出比较和逐题退化。暂不将过去 100 页候选重排的 62.22% 称作全库成绩。

`python -m paper_agent.cli colqwen-full-search --help` 提供普通文本的独立全库页面检索入口；它要求已封存的完整索引以及上述经 Chroma 逐页校验的无损缓存，逐批对全部页面计算 MaxSim，返回论文、页号、原图哈希和分数，不读取文本或 CLIP 候选。完整索引和缓存尚未齐备时该入口不能用于本次全库查询；小规模合成回归仅验证遍历、排序与来源守卫。
