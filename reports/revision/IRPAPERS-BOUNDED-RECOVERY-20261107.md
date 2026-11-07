# IRPAPERS 有界检索、页面切片与 ColQwen2 恢复

**本轮结果：** 已对 210 个固定问题实际测量有界 Chroma 查询；顶部五页与原精确基线一致，但有一条视觉开发问题的目标跌出前 20。八题开发切片探针没有证明视觉增益。ColQwen2 基础权重和兼容环境已恢复，真实 GPU 前向、两页 Chroma 写入与候选内 MaxSim 已跑通；完整候选基准尚未完成。基线和测试集汇总此前已曝光，不属于严格盲测。

## 下载事故、当前状态和依赖

开始本轮时，原 Xet 下载目录的两个 `.incomplete` 分片分别有 **768,000,000** 和 **335,389,810** 字节，均未完成且未通过整文件哈希。尝试用官方 Xet 在该目录续传时，客户端将它们重新置为 0；发现后立即停止，停止后原目录分别显示 0 和 **268,289,204** 字节。**原来约 1.10 GB 的断点没有完整保住，也没有证据表明可以从 Xet 缓存恢复；这是本轮操作造成的损失。** 原目录及当前残留均保留，此后未再对其发起写入。

新建目录 `models/colqwen2-base-recovery-20261107` 的单独 Xet 尝试长时间无可见写入，已停止并保留。此前 Xet 日志显示官方 CDN 连接出现 `tls handshake eof` 或 `IncompleteMessage`。本轮只读网络探针发现：通过本机现有代理，官方 Hugging Face 文件的两个连续 8 MiB HTTPS Range 均正确返回 206 和对应 `Content-Range`；直接连接 Hugging Face 文件失败，PyTorch 官方下载域的小范围请求通过代理和直连均成功。全过程保持证书验证，未使用第三方镜像。

因此在**另一个全新目录** `models/colqwen2-base-range-20261107` 使用 `scripts/download_colqwen_ranges.py` 通过官方 HTTPS 分段下载：每 16 MiB 段独立记录哈希、严格验证范围及长度；重启时重新核对已完成分段，未完成文件不覆盖或删除。经过多次有时限批次、一次 TLS 错误后的低并发续传，527 个分段均完成；两个最终分片长度及官方 SHA-256 均已校验通过。分段记录保留，最终权重与旧目录独立。

适配器 revision 为 `2b6ac8fb37f46a49e4841e599583d00ae8a20117`，其权重 SHA-256 `fc856312174dc99a4c7f88a2c54d9590a3b3f5b5a86e2728d7138c7f4758c6d5`；基础 revision 为 `9fe8a713422a7cb4ef79ca77a09b381ee2243101`。两片官方 SHA-256 分别为 `5afb3798d74668030808362aa4a4870cf71a0cfd1780287c9c55e40fb24e51d4` 和 `3da90e07b7b0c19944ee2cdcf537fa62ba5b01a9170b193d6b68353e9b033ec3`。这些固定值在下载和 smoke 脚本中核对。

从本机 `.venv-mm-20261030` **复制** PyTorch `2.6.0+cu124`、torchvision `0.21.0+cu124` 和 sympy `1.13.1` 到新 `.venv-colqwen-20261106`，未在旧环境安装或替换包；该新环境还使用 `colpali-engine 0.3.12`、transformers `4.53.3`、peft `0.16.0`。`pip check` 输出 `No broken requirements found`，CUDA 可用，指定模型的小样本前向已成功。复制 sympy 时 `__pycache__` 目录出现已存在提示；对非缓存的 1,546 个源文件逐个核对 SHA-256，均与目标一致。临时增加的旧环境引用 `.pth` 已改名为 `.pth.disabled` 并保留，新环境不依赖该引用。

## 真实 Chroma 近似与精确对照

新增独立 `paper_agent.bounded_search.search_grouped` 及可选的 `BoundedPageRetriever` 包装：按候选深度渐进查询、按可信 source 映射过滤论文／页面、按 `(paper_id,page_id)` 去重后分页；候选不足时显式返回 `candidate_depth_limit`，可选精确回退并记录路径。满足结果数量也只标记为近似，不能推断全局精确。对不完整、重复或分数乱序的候选列表现在直接报错。现有默认 Retriever 和冻结基线未修改。

开发集实际比较 100、300、600 个片段，选择最小的深度使两组开发题 Hit@5 不低于精确 BGE，且平均页面 Top-5 集合重合至少 99%；冻结深度为 300，计划 `runs/irpapers-bounded-20261107/plan.json` SHA-256 `0f8aae1998b2513ec206214eff4efb6f86e995174cbcb6be35999c893eaf20b4`。冻结后在全部 210 个原问题执行；原始逐题 `results.jsonl` 和汇总同目录。后来增加了拒绝不完整/重复/乱序候选的守卫，原实验执行时的代码已另存同目录 `bounded_search.snapshot.py`；不能声称新增守卫参与了原实验。守卫后的 157 条开发题重新运行见 `runs/irpapers-bounded-guard-20261107/`：300 和 600 深度的逐题 Top-20 与原开发运行相同，100 深度有四条 Top-20 尾部不同，选出的深度仍为 300；没有重新运行保留题。

| 集合 | 方法 | Hit@1 | Hit@5 | Hit@20 | MRR 或 MRR@20 |
|---|---|---:|---:|---:|---:|
| 标准 180 | 精确 BGE | .4000 | .7333 | .8611 | .5487 全排序 MRR |
| 标准 180 | 有界 BGE 300 | .4000 | .7333 | .8611 | .5466 MRR@20 |
| 视觉 30 | 精确 BGE | .2000 | .3333 | .5000 | .2810 全排序 MRR |
| 视觉 30 | 有界 BGE 300 | .2000 | .3333 | .4667 | .2739 MRR@20 |
| 标准保留 42 | 有界 BGE 300 | .4048 | .7143 | .8333 | .5537 MRR@20 |
| 视觉保留 11 | 有界 BGE 300 | .3636 | .4545 | .6364 | .4139 MRR@20 |

MRR@20 将 20 名之后截断为零，不能与全排序 MRR 当作同一指标比较。210 条的 Top-5 顺序均与精确基线一致，但 20 页集合并非全部一致；视觉开发 `visual-008` 的目标从精确 Top-20 掉出，不能声称无召回损失。冲突标签继续按未知计入分母，空转写案例未排除；这些是页面检索，不是图表区域或引用验证。BM25、文本混合、整页 CLIP 与等权融合的 180/30 全集及开发／保留基线见 [冻结基线报告](CHROMA-IRPAPERS-20261106.md)，本轮未将有界 BGE 的独立页面结果冒充混合模型的完整评测。

仅向量查询及页面分组：标准题中位约 7.97 ms、p95 11.23 ms；视觉题中位约 8.45 ms、p95 12.41 ms。10 条精确全量查询的首条约 711 ms，缓存后中位约 9.88 ms；两者返回深度不同，不能当作同工作量的端到端速度对比。独立进程的 RSS 在有界查询后约 2.66 GB、精确缓存后约 2.71 GB；包含 BGE 模型和运行时。批量编码 210 问约 4.49 秒，GPU 峰值分配约 1.15 GB；原文本索引约 80.64 MB。Chroma 打开时仍按批读取所有向量核对指纹，不能说启动阶段不读取全量数据。

新增包装器的实际端到端密集页面检索另测了三条开发问题：启动加载约 40.15 秒，首问约 3.59 秒、后两问约 0.20 和 0.17 秒，Top-5 顺序均与精确基线一致；这三次包含查询编码及有界检索，但不包含 BM25、融合或生成，GPU 峰值约 1.15 GB、进程最终 RSS 约 2.56 GB。具体记录 `output/irpapers-bounded-live-probe-20261107.json`；三例不能代表生产延迟分布。

对三个真实开发问题，固定一个与 gold 无关的论文 `irp-98`，分别测其前 5 页、下一页及固定三页的范围过滤；9 次排序均与精确结果一致，8 次因候选不足而显式回退到全量精确搜索。完整结果 `output/irpapers-scope-probe-20261107.json`，不能推广为任意范围都无需回退。

## 固定 CLIP 页面切片探针

冻结前四个标准和前四个视觉开发案例；候选是每题全库文本前 10 与全库整页 CLIP 前 10 的页面并集，不使用 gold、答案或论文过滤。111 个不同页面各切为顶部和底部 60%（20% 重叠），白色补齐为正方形，保留原页面哈希及像素框，然后用同一固定 CLIP 图像快照在 GPU 实际编码 222 张切片。计划 SHA-256 `a679975e09842dcb09808430a05cab93ae834db8f34f68a5ff04d6acac1512ca`；结果与来源清单位于 `runs/irpapers-tile-probe-20261107/encoded/`，独立 Chroma 索引 `indexes/irpapers-clip-tiles-probe-20261107`，约 3.50 MB。编码及查询共约 26.63 秒，GPU 峰值约 1.17 GB，进程最终 RSS 约 1.46 GB。

八题候选覆盖 5/8：标准 4/4，视觉 1/4；只在候选内按每页**最高一个切片相似度**排名，目标 Top-5 为标准 2/4、视觉 0/4。文本加切片 RRF（视觉权重 .25）为标准 4/4、视觉 1/4，与原文本 Top-5 在这八题相同。三条视觉目标未进入候选时，切片重排无法挽回。该探针不是 3,230 页全库切片索引，也不是 210 题评测；没有观察到足够证据证明多模态提升，故未修改默认视觉分支。

之后重新编码八个问题并实际查询该 222 切片的 Chroma 索引，八次的切片 Top-5 和 Top-20 集合均与直接余弦排序一致，搜索路径均为 HNSW；加入来源和模型哈希核对后的结果在 `output/irpapers-tile-chroma-check-v2-20261107.json`。首次受限运行因系统临时目录权限失败，改用工作区临时目录后成功。此核验不扩大候选语料，也不将切片当作可引用的图表区域。

## ColQwen2 可恢复存储与剩余工作

新增 `paper_agent.resumable_colqwen.ResumableColQwenStore`：建立独立 Chroma collection，将每页的论文、页面、图像 SHA-256、token 序号、模型／语料契约和向量哈希保留；写入前持久记录该页意图，按批 upsert，回读确认后才提交。重启只允许相同输入续写；有未完成页、孤儿向量或契约不符时不允许封存。封存后查询使用逐 query token 对页面 token 的真实 MaxSim 算法，不使用均值。真实 Chroma 的合成向量测试模拟了 513 token 页面在第二批前中断、重开恢复、来源不符拒绝及封存；它不是模型性能结果，也未实测真实模型的大规模内存、磁盘、吞吐或中断恢复。

真实 smoke `runs/colqwen-real-smoke-20261107/` 以现有文本排序的两个候选页 `2_1`、`2_2` 和 `standard-006` 问题，在 RTX 4060 `cuda:0`、256 视觉 token 上限下实际运行：查询 33×128，两页分别 258×128；attention mask 移除 padding，MaxSim 与官方 processor 逐页一致。全程包括权重哈希和加载约 104.37 秒，GPU 峰值 4,651,127,296 字节，进程最终 RSS 1,754,161,152 字节；GPU 能运行该设置。两页实际 MaxSim 分别约 20.7737、21.7567，第二页排在标注目标第一页之前，成功运行并不证明检索正确。

随后的增强检查最初因适配器文件的旧键名和当前模型键名不一致而拒绝，失败记录保留在 `runs/colqwen-real-verified-20261107/` 与 `runs/colqwen-adapter-diagnostic-20261107/`。核查确认 PEFT 按模型的官方转换规则 `^model → language_model` 加载旧键；使用相同官方转换重新逐张量核对后，基础模型缺失／多余键均为 0，394 个适配器张量与固定快照的最大绝对差为 0。增强 smoke `runs/colqwen-real-audited-20261107/` 成功，向量与首次结果的 SHA-256 一致；GPU 峰值约 4.65 GB。该结果验证固定权重加载及两页计算，仍不证明整体召回质量。

首次真实向量导入触发了原有字节级校验缺陷：Chroma 回读存在最大约 `2.98e-8` 的浮点差，且后续索引更新后少量位可再次变化。原始失败索引保留并用于真实续建。修复后逐页保存输入向量的独立哈希参考，并在读取时要求 Chroma 值与该参考逐元素相差不超过 `1e-6`；参考文件及来源哈希独立校验，这允许该容差以内的变化而非声称逐位一致。恢复写入后 516 个真实 token 成功封存，Chroma 中重排得分与模型输出一致，见 `output/colqwen-real-smoke-import-20261107.json`。另建全新两页索引 `indexes/colqwen-real-smoke-clean-20261107`，导入并封存约 0.35 秒、两页重排约 0.021 秒、索引约 1.82 MB，见 `output/colqwen-real-smoke-clean-20261107.json`；这不是大规模性能估计。

该索引又在新进程中重新打开并完成相同的两页重排，结果 `output/colqwen-real-smoke-reopen-20261107.json`；此次两页查询约 0.307 秒，显示小样本时延受冷启动和系统负载影响明显。

对同一开发问题的 98 个已提交候选，逐页读取和 CPU MaxSim 实测约 12.55 秒，目标第 2、文本基线第 1。随后将同一公式按 token 批量放在 GPU 计算，全部 98 页分数与逐页参考最大差 `3.88e-6`，Top-20 顺序一致；该次读取约 5.90 秒、批量计算约 0.20 秒、总计约 6.12 秒，GPU 峰值分配约 25 MB。记录 `output/colqwen-dev-batched-verified-20261107.json`。完整评测会使用封存后可校验的页面和有界缓存，实际全量延迟仍待测量。

冻结的文本前 50＋CLIP 前 50 候选覆盖及限制见 [前一报告](IRPAPERS-RECALL-COLQWEN-20261106.md)。完整候选模型编码、查询向量和 180＋30 问评测尚未执行，真实视觉增益、引用正确性及大规模运行资源仍 unknown。

另外准备了 `scripts/colqwen_encode_candidates.py`、`scripts/colqwen_patch_provenance.py`、`scripts/colqwen_candidate_eval.py`、`scripts/colqwen_freeze_eval.py` 和 `scripts/colqwen_combine_eval.py`：它们分别保存可校验的逐页／逐查询模型输出、记录处理器图像 token 顺序与网格、可恢复导入 Chroma 并仅在冻结候选内做 MaxSim、在开发结果完成后冻结测试配置，以及校验后汇总两部分。完整候选索引封存前须将页面网格、有效 token 位置、原图哈希和编码检查点关联写入并绑定清单哈希；不声称已得到像素级引用框。测试集编码必须提供与开发结果哈希及模型设置一致的冻结文件。尾部页面沿用原文本排名，不能描述为全库 MaxSim。

真实开发集续建已开始：冻结候选覆盖 2,114 个不同页面，首批生成全部 157 个开发问题向量及 20 页向量；首批本地页面编码中位约 0.320 秒、问题编码中位约 0.058 秒，整个首次批次含模型加载约 33.29 秒、GPU 峰值约 4.65 GB、进程最终 RSS 约 1.89 GB。后续运行已从检查点继续；一个增量导入时点已有 29 页、7,522 个真实 token 在独立 Chroma 中，索引约 13.95 MB，采样 RSS 约 370 MB；该导入报告明确标记 `incomplete`。这是早期进度快照，非完整评测或大规模性能保证。

后续开发索引已写入并封存 2,114 页、550,217 个真实 token 向量，页面网格与 token 顺序清单已绑定哈希；候选内逐题计分只完成部分检查点，尚无完整开发集或保留集结果。最新阶段核对发现继续 ColQwen2 计分偏离该会话中用户限定的范围，已停止并保留检查点，详见 [第一阶段状态](MULTIMODAL-PHASE1-STATUS-20261107.md)。以上进度不能作为整体召回率或大规模性能结论。

## 复现及安全恢复

使用项目根目录；输出必须是新的目录或文件。下载脚本对现有分段做完整校验后续传，每批上限 300 秒，失败文件保留。只有 `status: complete` 且两个整片 SHA-256 均匹配时才可进入 smoke；大文件需额外约两倍空间以保留校验分段及最终分片。

```powershell
$c = '.\.venv-colqwen-20261106\Scripts\python.exe'
$p = '.\.venv-gpu\Scripts\python.exe'
& $c -m pip check
& $c scripts/download_colqwen_ranges.py --directory models/colqwen2-base-range-20261107 --workers 6 --max-seconds 300
# 按需重复上一条，直到 status 为 complete；不要重新使用原 Xet 目录。
& $c scripts/colqwen_smoke.py --base models/colqwen2-base-range-20261107 --adapter models/colqwen2-v1-20261106 --output runs/colqwen-real-smoke-NEW --device cuda:0 --max-visual-tokens 256
& $p scripts/colqwen_import_smoke.py --smoke runs/colqwen-real-smoke-NEW --index indexes/colqwen-smoke-NEW --output output/colqwen-smoke-import-NEW.json
& $p scripts/irpapers_scope_probe.py --output output/irpapers-scope-NEW.json
& $p scripts/irpapers_bounded_live_probe.py --output output/irpapers-bounded-live-NEW.json
& $p scripts/irpapers_tile_probe.py choose --baseline runs/irpapers-chroma-baseline-v2-20261106 --output runs/irpapers-tiles-NEW/plan.json
& .\.venv-mm-20261030\Scripts\python.exe scripts/irpapers_tile_probe.py run --baseline runs/irpapers-chroma-baseline-v2-20261106 --plan runs/irpapers-tiles-NEW/plan.json --output runs/irpapers-tiles-NEW/encoded --device cuda:0
& $p scripts/irpapers_tile_import.py --encoded runs/irpapers-tiles-NEW/encoded --output indexes/irpapers-tiles-NEW
```

真实 smoke 和导入均成功后，可按以下顺序执行候选评估。编码器及索引命令会报告 `paused` 或 `incomplete`，以相同契约和输出目录重复各自命令可续建；评估结果文件和测试配置必须使用新路径。OOM 或兼容性错误须保留错误并另建不同设置的运行，不能把新参数静默写入旧契约。

```powershell
$base = 'models/colqwen2-base-range-20261107'
$adapter = 'models/colqwen2-v1-20261106'
$smoke = 'runs/colqwen-real-smoke-NEW'
$candidates = 'runs/irpapers-colqwen-candidates-20261106'
$baseline = 'runs/irpapers-chroma-baseline-v2-20261106'
& $c scripts/colqwen_encode_candidates.py --base $base --adapter $adapter --smoke $smoke --candidates $candidates --baseline $baseline --partition dev --output runs/colqwen-dev-NEW --max-seconds 600
& $c scripts/colqwen_patch_provenance.py --encoded runs/colqwen-dev-NEW --adapter $adapter --output runs/colqwen-dev-NEW/patch-provenance
& $p scripts/colqwen_candidate_eval.py --encoded runs/colqwen-dev-NEW --candidates $candidates --baseline $baseline --patch-provenance runs/colqwen-dev-NEW/patch-provenance --index indexes/colqwen-dev-NEW --output runs/colqwen-dev-NEW/results.jsonl --max-pages 100
& $p scripts/colqwen_freeze_eval.py --development-summary runs/colqwen-dev-NEW/results.jsonl.summary.json --encoding-contract runs/colqwen-dev-NEW/contract.json --output runs/colqwen-selection-NEW.json
& $c scripts/colqwen_encode_candidates.py --base $base --adapter $adapter --smoke $smoke --candidates $candidates --baseline $baseline --partition test --selection runs/colqwen-selection-NEW.json --output runs/colqwen-test-NEW --max-seconds 600
& $c scripts/colqwen_patch_provenance.py --encoded runs/colqwen-test-NEW --adapter $adapter --output runs/colqwen-test-NEW/patch-provenance
& $p scripts/colqwen_candidate_eval.py --encoded runs/colqwen-test-NEW --candidates $candidates --baseline $baseline --selection runs/colqwen-selection-NEW.json --patch-provenance runs/colqwen-test-NEW/patch-provenance --index indexes/colqwen-test-NEW --output runs/colqwen-test-NEW/results.jsonl --max-pages 100
& $p scripts/colqwen_combine_eval.py --dev runs/colqwen-dev-NEW/results.jsonl --test runs/colqwen-test-NEW/results.jsonl --selection runs/colqwen-selection-NEW.json --output runs/colqwen-combined-NEW.json
```

最新完整回归为 **454 passed、1 skipped、2388 warnings**，包括包装器、检查点及中断恢复、图像 token 清单绑定、批量 MaxSim、开发配置冻结、候选完整性、分段下载守卫、文本与记忆隔离测试。`git diff --check` 单独执行。大部分新增 warnings 来自 Chroma 的 Pydantic 废弃提示。

本轮不发起远程生成模型请求，新增请求数和 token 均为 0。最终下载进度以本次执行记录为准；任何生成模型消耗仍须遵守此前独立预算。
