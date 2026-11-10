# IRPAPERS 纯 ColQwen2 全库评测（2026-11-10）

本记录与[先前候选内评测](COLQWEN-CONTINUATION-20261108.md)分开。目标是对每个文字问题在 3,230 张原始页面上计算真实 MaxSim：`sum_i max_j (query_i · page_j)`，只使用 attention mask 中的有效 token；不使用 BGE、BM25、CLIP 或 gold 筛选页面，也没有进行多向量压缩。本次语料的页面转写是 benchmark transcription，不是 Docling PDF 原文；图像结果仅为页面级定位，不能作为图表 bbox 或问答引用验收。

## 数据与真实向量

本轮重新计算原始页面、标准题和视觉题 CSV 的 SHA-256，分别为 `8fe316290b55c71bd1641603b5feb02117672028f7d1af45bcad1c111c53611b`、`a87d8263d8b498cdaceb6cb719abcd0a584ded3993910aad41edab1db8fa6ed3` 和 `972aa9f2f1541defbefc4226667dba04653af8be9c8ad5351c201fdc82b89b93`；页面清单 SHA-256 `d127db27db1077edf0f19defc7a643b3071c33cfe56acd0502b40d3a748faac9`。旧审计为 3,230 页、166 篇、标准 180 题和视觉 30 题；视觉题与标准题重叠 13 篇论文、19 个目标页，不能称独立测试集。标准题 `standard-143` 的页面标识冲突保留为 unknown 并计入分母；`standard-090` 目标页转写为空，仍保留。历史保留集汇总已经曝光，不是严格盲测。

开发和保留页面编码契约的模型 revision、两片基础权重、适配器哈希、128 维、256 视觉 token 上限、处理器版本、设备和源页面清单一致。1,361 个重叠页面逐一比较向量（绝对容差 `1e-6`）、图像哈希、网格、有效 token 位置和来源，均通过；2,322 页复用已验证模型向量，其余 908 页用同一模型与隔离环境真实编码。当前运行的处理器配置与先前记录的哈希一致；历史检查点没有绑定所有配置文件哈希，不能追溯性地声称已验证历史每个配置文件。

页面编码检查点沿用项目已有 `.npz` 无损文件压缩，解码后保留所有 float32 token；这不是降维、池化或近似检索。Chroma 保存各 token，计分不使用 Chroma 的近似向量搜索。
3,230 个页面检查点共 841,060 个有效 token，每页 258–263 个；封存索引逐 token 回读也核对了这些总数。

本轮逐个校验 210 条查询检查点的题文哈希、向量文件哈希、维度、有限值与归一化；有效 token 数 27–66。209 个标签一致问题的目标页全部在 3,230 页来源清单中；冲突题的两个可能页面也都存在，但目标仍为 unknown。该覆盖核查不参与在线候选筛选，也不是检索命中结果。核查文件 SHA-256 `7e845e211a2c763de80c1d05130d405f64b20ea30070211bf532ff0c871f768f`；模型／运行时契约复核 SHA-256 `4ec71f1217be48d02758fe78e2d3f57ad6b90d52d848eb2c6cda386b5cdaab50`。

三张不依赖 gold 选择的真实页面和一个真实开发问题上，GPU 批量 MaxSim、独立逐页计算与官方 processor 的最大差 `1.0431e-6`，原始输出 `output/colqwen-full-probe-20261110.json`。向量、有效 token 和页面来源保留；不将图像 token 当作像素框。

再从两套旧候选索引均未覆盖的 908 个新编码页面中按来源顺序取前三页，与同一真实问题进行相同数值对照，三种实现最大差 `1.9074e-6`；页面选择不依赖 gold。结果 `output/colqwen-full-new-pages-probe-20261110.json` SHA-256 `a80f535ab96e1f78526cd3881152ed50b71e22c1455a5010bc9794ac09353829`。两次探针均只验证数值实现，不是召回率。

## 全库结果

210 道问题均完成对全部 3,230 页的精确 MaxSim 打分；先冻结运行契约，在 157 道开发题上选择页面级 RRF（文本权重 1、ColQwen 权重 0.25，k=60），然后才运行 53 道保留题。选择规则要求标准题、视觉题开发集 Hit@5 不低于文本基线，满足条件时最大化两组 MRR 之和。保留集历史汇总已曝光，且视觉题与标准题页面重叠，以下是复现实测而非严格盲测。标签冲突的 1 题保留在分母并计为未命中。

| 题组与方法 | Hit@1 | Hit@5 | Hit@20 | MRR |
| --- | ---: | ---: | ---: | ---: |
| 标准 180：BM25 | 45.00% | 77.78% | 89.44% | 0.5911 |
| 标准 180：稠密文本 | 40.00% | 73.33% | 86.11% | 0.5487 |
| 标准 180：文本融合基线 | 39.44% | 77.22% | 91.11% | 0.5606 |
| 标准 180：CLIP | 0.56% | 2.78% | 6.67% | 0.0187 |
| 标准 180：文本+CLIP | 8.33% | 30.00% | 82.22% | 0.2075 |
| 标准 180：旧候选内 ColQwen | 27.78% | 62.22% | 87.78% | 0.4405 |
| 标准 180：纯全库 ColQwen | 23.89% | 55.56% | 75.56% | 0.3860 |
| 标准 180：文本+全库 ColQwen | 40.56% | 78.33% | 92.78% | 0.5661 |
| 视觉 30：BM25 | 13.33% | 40.00% | 50.00% | 0.2286 |
| 视觉 30：稠密文本 | 20.00% | 33.33% | 50.00% | 0.2810 |
| 视觉 30：文本融合基线 | 16.67% | 40.00% | 60.00% | 0.2794 |
| 视觉 30：CLIP | 0.00% | 0.00% | 3.33% | 0.0095 |
| 视觉 30：文本+CLIP | 10.00% | 23.33% | 50.00% | 0.1915 |
| 视觉 30：旧候选内 ColQwen | 20.00% | 40.00% | 53.33% | 0.3041 |
| 视觉 30：纯全库 ColQwen | 20.00% | 40.00% | 50.00% | 0.2685 |
| 视觉 30：文本+全库 ColQwen | 23.33% | 40.00% | 60.00% | 0.3291 |

| 分区 | 数量 | 文本 Hit@5 / MRR | 纯全库 Hit@5 / MRR | 融合 Hit@5 / MRR |
| --- | ---: | ---: | ---: | ---: |
| 标准开发 | 138 | 77.54% / 0.5443 | 50.72% / 0.3498 | 78.26% / 0.5450 |
| 标准保留 | 42 | 76.19% / 0.6142 | 71.43% / 0.5046 | 78.57% / 0.6353 |
| 视觉开发 | 19 | 31.58% / 0.2122 | 36.84% / 0.1940 | 31.58% / 0.2537 |
| 视觉保留 | 11 | 54.55% / 0.3954 | 45.45% / 0.3973 | 54.55% / 0.4593 |

与文本基线逐题比较，纯全库在标准题 Hit@5 新增 11 个、退化 50 个；视觉题新增 5 个、退化 5 个。融合在标准题新增 9 个、退化 7 个，视觉题新增 2 个、退化 2 个。旧候选集覆盖标准题 gold 95.00%、视觉题 gold 76.67%；覆盖纯全库前 20 页的比例分别为 46.72%、30.67%，因此候选内排名并不等同于全库排名。全部逐题排名、命中变化和分区统计见 [轻量结果文件](colqwen-full-results-20261110.json)，完整 3,230 页排名、分数和前 20 页来源在 `runs/colqwen-full-eval-20261110/results-{dev,test}.jsonl`；其 SHA-256 分别为 `5b62e96b1af7a291ad5f02056d97c2bef52602c56060b13542bfee772f3741a6`、`9c50f65b1174e42f098aaec8af0cf4a3e189acdfcb3672df7075c375b7b6ba8d`。冻结选择 SHA-256 `e18ce6f9b79be6b06801de19dff4c77c6c9ef44fc541e55401313c3d88632e18`，完整比较 SHA-256 `0453a8082306d0118a36989a3c4cb789a000eaa91b5a92622aeda1a1d09d426d`。

## 时间与资源

在 RTX 4060 Laptop GPU（8,188 MiB）上，开发阶段打开索引 2.28 秒，建立并核验缓存 87.66 秒，整个进程 206.58 秒；保留阶段重新打开 3.87 秒、核验缓存 47.83 秒、整个进程 87.16 秒。开发进程采样最高 RSS 2.57 GB；打分进程报告的 GPU 张量峰值约 21.6 MB，不含独立模型编码进程及其他显存占用。标准题热缓存全页打分中位数 0.707 秒、p95 0.819 秒，其中页面读取中位数 0.078 秒、批处理计算中位数 0.620 秒；视觉题全页打分中位数 0.719 秒、p95 0.829 秒。历史查询编码检查点耗时中位数约 0.058 秒，来自另一次已加载模型的运行，不能当作冷启动时间或与本轮计时相加后声称端到端时延。

单独运行真实 `standard-006` 查询，启动到结束 68.28 秒，其中查询编码子进程含模型加载 58.31 秒、全页打分及来源校验 5.12 秒；前 20 页与开发集检查点完全一致。单次冷启动测量不代表稳定吞吐。之前候选方案和本轮分量来自不同运行，不能据此直接比较在线延迟或成本。

完整测试首次运行 469 通过、1 跳过、1 失败：一项与本次检索无关的本地审计测试在 Windows 原子替换临时文件时遇到 `WinError 5`，原因未确定。该文件的 11 项测试在新临时目录全部通过；再以新临时目录完整重跑为 470 通过、1 跳过（44.00 秒）。

## 进度与恢复

3,230 页向量已齐备在 `runs/colqwen-full-20261110`；独立 Chroma 索引 `indexes/colqwen-full-v2-20261110` 已写入并封存 3,230 页、841,060 个 token，封存前逐 token 批量回读与原始向量参考核对。封存文件 SHA-256 `77d8f33b0b5ad9d1c2a95dac6aa06617f05bbb0b4d7b1145829e4d92c34ca487`；索引约 1.553 GB，导入进程计时约 1,717.50 秒、结束 RSS 约 1.21 GB（并非采样峰值）。每页写入记录、输入参考和来源单独保存；中断时以相同命令续建，不覆盖原索引或检查点。若复制中断留下无 marker 的不完整向量文件，先保留并改名该新运行目录中的孤儿文件，再以同一契约续跑；不要删除原数据或旧检查点。

导入启动后为加速后续重启和回读，工作区中的导入脚本增加了批量封存实现；已运行进程继续使用启动时的原实现。启动版本按本次记录的代码改动反向重建并保存在 `runs/colqwen-full-20261110/importer-start-reconstructed.py`，SHA-256 `a3809b1671713db550fa54bc50b61f4fe1849bf76025925584c974af27a38d61`；启动时没有预先保存该文件或哈希，因此这是重建版本，不冒充当时的预先冻结快照。若未来恢复并封存原索引，应在新进程重新回读核验；本次评测使用的是另建并完成全量回读的 v2 索引。

为启用批量封存，在 1,799 页已提交、1,800 页有写入意图时中断原进程；两次尝试恢复都在未完成页面附近退出且没有 Python 堆栈，第二次明确返回 Windows 原生访问异常 `0xc0000005`，第一次仅记录到退出码 1，原因尚未确定。原索引 `indexes/colqwen-full-20261110`、待完成页面和诊断日志均保留，不能称其已恢复或封存。新建 v2 目录从已验证向量重新导入；批量写入的中断续写、来源篡改和封存守卫已用真实 Chroma 的合成向量回归，这不是模型效果结果。

v2 启动后立即保存了其源码快照：导入脚本 SHA-256 `c683b8ae0b44ba1c3bab5449f286d452fe5a5827419b3c3566270a62eb77020b`，批量存储实现 SHA-256 `2b25dccfab1fa51064ca5ee2cece8f515c10bed74886eea24d8a28efa7795388`；快照和日志位于全库运行目录。另在既有封存索引上真实批量读取 16 页、4,178 个 token，与逐页读取最大差为 0；该检查不衡量检索效果。908 页新增编码的首末检查点时间间隔约 342 秒，不含首个编码前的模型加载；从全库编码契约创建到最后一个新页检查点约 566 秒，包含复用核验和模型加载。时间由文件时间推算；全轮显存峰值未独立采样，不能将历史烟测峰值直接当成本轮实测。

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

`python -m paper_agent.cli colqwen-full-search --help` 提供普通文本的独立全库页面检索入口；它要求已封存的完整索引以及上述经 Chroma 逐页校验的无损缓存，逐批对全部页面计算 MaxSim，返回论文、页号、原图哈希和分数，不读取文本或 CLIP 候选。本轮缓存已完成并用于真实查询对照；小规模合成回归另外验证遍历、排序与来源守卫。
