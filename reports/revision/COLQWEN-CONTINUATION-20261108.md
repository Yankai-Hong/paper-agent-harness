# ColQwen2 独立分支续跑记录（2026-11-08）

## 研究检查点和范围

主线冻结提交 `d99827684e5f9a2b710d5133b219b78a546b77ae`；分支 `codex/colqwen2-visual-retrieval` 最初从研究检查点 `fb8069a3cdc62b0d3ea2ce78e2b8cd9790c30106` 建立，主线页面检索完成并回归后，本任务的分支提交整理到该冻结提交之上。主线完整回归 **459 passed、1 skipped**（38.29 秒）；受限连接下此前七项本机 HTTP 测试失败，允许本机回环连接后完整通过。冻结覆盖可复现的文本及页面检索，**不是端到端问答和引用质量全部验收完成的声明**：页面接口尚未替换 `ask`／`chat` 的默认证据选择，真实 PDF 的全面引用质量仍需验收。

继承 [IRPAPERS 基线](CHROMA-IRPAPERS-20261106.md) 的数据及全部页面范围和 [恢复记录](IRPAPERS-BOUNDED-RECOVERY-20261107.md) 的模型与候选协议。本轮重新计算三个原 CSV 的 SHA-256，与原审计一致：页面 `8fe316290b55c71bd1641603b5feb02117672028f7d1af45bcad1c111c53611b`、标准题 `a87d8263d8b498cdaceb6cb719abcd0a584ded3993910aad41edab1db8fa6ed3`、视觉题 `972aa9f2f1541defbefc4226667dba04653af8be9c8ad5351c201fdc82b89b93`。准备后页面清单 SHA-256 `d127db27db1077edf0f19defc7a643b3071c33cfe56acd0502b40d3a748faac9`；基线冻结文件 SHA-256 `697b917a1e852ffc3845cf90b5d338ca1a2242a9f7486648c91cbbca0f313d13`；候选计划 SHA-256 `dbea772bc28840b5f236839ec88a46c4c64b667b92bf164b7878e29c60e3f32d`。先前测试集汇总已曝光，不是严格盲测。转写是 benchmark transcription，不是 Docling 原文，不能提供像素级 bbox。

已有开发索引封存 2,114 页、550,217 个 token，`sealed.json` SHA-256 `e91ae02916774034adfd3de616c667c1f8195a735e011dc787fbf51fa1b97e45`，当前目录约 1,017,400,350 字节；编码契约 SHA-256 `109bf2fb728623e2785611a816ae43448407d7e396a3ee0a0486d6e35b923f11`。固定适配器和基础模型 revision、各权重哈希及 256 视觉 token 设置见恢复记录和契约。隔离环境 `pip check` 通过。此索引只覆盖开发题候选页面，不是全语料页索引。

## 查询接口和真实探针

`ColQwenCandidateRetriever` 接收普通文本和调用者提供的页面候选，在独立多向量 collection 中按真实 MaxSim 排序。来源清单哈希绑定索引，检验页面 ID、论文、原图哈希、页码、模型版本和权重契约；缺页或不匹配直接报错。每个结果只声明页面级定位，不把整页当作图表 bbox，也不声称同页文字与图像互相支持。查询模型通过隔离进程执行，attention mask 排除 padding；候选生成与问答管线还未接入该接口，不能将其称为完整 Docling 问答。

在 RTX 4060 8 GB 上，`standard-006` 的候选表前两页（与 gold 无关）执行真实模型查询：33×128 查询向量与原模型检查点最大绝对差为 0；两页排序为 `2_2`、`2_1`，MaxSim 约 21.7567、20.7737。带清单校验的复验包含进程启动、权重校验和模型加载约 36.35 秒，随后两页读取和计分约 6.44 秒；另一冷启动探针约 67.77 秒。它们是少量单次测量，且目标页排第二，不代表总体效果或延迟分布。结果文件 `output/colqwen-live-query-manifest-20261108.json` SHA-256 `c01dd3d66539514b3bbd685260415d6829c76afc14292e6b2dfe7bf221be13a8`，为忽略资产。

```powershell
$p = '.\.venv-gpu\Scripts\python.exe'
& $p scripts/colqwen_live_probe.py --case standard-006 --index indexes/colqwen-dev-20261107 --encoded runs/colqwen-dev-20261107 --candidates runs/irpapers-colqwen-candidates-20261106 --baseline runs/irpapers-chroma-baseline-v2-20261106 --prepared data/raw/irpapers-prepared-20261106 --python .venv-colqwen-20261106/Scripts/python.exe --base models/colqwen2-base-range-20261107 --adapter models/colqwen2-v1-20261106 --output output/colqwen-live-query-NEW.json
```

现在还提供 `python -m paper_agent.cli colqwen-search-text --help`；它接收普通文本和由调用方提供的候选页面 JSON 数组，未索引页面会报错，不会静默省略。使用相同两页实际运行该命令成功，包含启动的总耗时 46.85 秒，输出 `runs/colqwen-cli-probe-20261108/stdout.json` SHA-256 `5b7cd230be70717fc040a5ec00b1d828fe7e033050339146abb638dccdbc7e9d`。它尚未接到 `ask`／`chat` 或自动候选生成，不能当作已完成普通问答接入。固定模型的当前配置与 tokenizer 哈希见 [本机配置清单](colqwen-config-hashes-20261108.json)；历史实验并未绑定这些配置文件哈希，因此不能据此证明历史配置内容。

新增的 `colqwen-hybrid-search` 使用现有 BGE+BM25 文本索引和独立 CLIP 页面索引，自动取各自前 50 页的并集；实验权重 `.05` 可融合候选内 ColQwen 前 20 页。它支持已绑定来源的 IRPAPERS 与 Docling 页面集合；索引缺少候选页时会报错。对 `standard-006` 的一次真实复核在同时导入保留索引时耗时约 165.18 秒；98 页候选**集合**与冻结基线相同，CLIP 前 50 和 ColQwen、融合前 20 顺序一致，文本前 50 中第 33、34 名交换，因此候选顺序不同。输出 `runs/colqwen-hybrid-live-v2-20261108/stdout.txt` SHA-256 `cfc0105a1732ce2d80efeb074d81e55a624c9ed5660c85b75cbaf0677a689d7b`。第一次运行因系统临时目录无权写入而失败，失败记录保留；第二次使用项目内临时目录。上述单例不代表实时延迟分布或实际问答、引用正确率。

之后增加保留所有文本候选原始证据 ID 和排名的输出，再对同一题实际复验，98 页集合和 ColQwen 前 20 仍与冻结结果一致，保留 9,166 个文本候选引用，耗时约 130.64 秒；输出 `runs/colqwen-hybrid-live-v3-20261108/stdout.txt` SHA-256 `62fe2370b34130673f4627fc3ae8120ecc1f112f4f45bb665fc0e89d96c17519`。上述两次真实复核运行时实验权重为 `.05`；保留组结果出现视觉题退化后，入口默认改为 0，复现实验必须显式传 `.05`。

## 真实 Docling PDF 页面探针

继承 [Docling+CLIP 页面记录](DOCLING-CLIP-PAGE-20261108.md) 的五篇 PDF、39 页、共享文本索引及 21 条标为未经核实的表格文本；不把它们和 IRPAPERS 转写混算。对这 39 张真实 PDF 渲染页实际运行固定 ColQwen2 GPU 编码，39 页共 10,062 个非 padding token，编码含模型加载约 38.73 秒，显存峰值约 4.65 GB，结束 RSS 约 1.63 GB。独立 Chroma 索引约 19.8 MB，封存哈希 `5b5b7afd1f699ab72549c7044e8dc7ba61b90bee3d701512cd1354dfdbdeb5b2`；新进程重开、逐页回读并核对官方处理器网格与 token mask，39 页全部通过，重开校验约 1.38 秒。另对实际问题和两张 PDF 页面将 MaxSim 与官方 processor 逐页对照，最大差 `3.28e-7`；记录 SHA-256 `163a8d27ef9e465c9d0b1d0b078ee150e6d11a3ea5219460f9b0f75096bf891b`。图片 token 位置不是像素 bbox。

两个已见开发问题均不施加目标论文或页码过滤；39 页都进入候选，因此对这小语料实际比较了全部 39 页，不能推广为 IRPAPERS 全库 MaxSim。表格问题的目标 `1909.12079` 第 5 页：文本第 1、CLIP 第 3、ColQwen 第 1、默认文本融合第 1，冷启动约 69.30 秒；返回独立表格文本、整页和表格图，原页与裁图代理复核见主线记录。输出 SHA-256 `021d3667560f545cbc9d896c1e00a796272a8e5e8e9b07e8be2cea51b3b672d6`。Figure 3 的 hints 问题目标 `1909.06708` 第 4 页：文本第 2、CLIP 第 9、ColQwen 第 6、默认融合第 2，冷启动约 65.40 秒，返回独立 Figure 3 证据；输出 SHA-256 `3b65b9fba1d0fcc6d97c7733aacd139748f3ff263b790cc3fa469586482597ae`。同页证据均标记未验证语义关联，表格文本的解析状态仍为 unknown；原页代理复核不等于人类金标。没有在本轮发起远程生成问答，`ask`／`chat` 尚未接入此页面接口。

按主线报告准备页面和共享文本索引后，可用以下命令在新路径重建 PDF 的 ColQwen 页面向量和独立索引；两种索引使用同一页面清单但维度和 collection 完全独立。原始页面／向量检查点与模型不会覆盖。

```powershell
$p = '.\.venv-gpu\Scripts\python.exe'
$c = '.\.venv-colqwen-20261106\Scripts\python.exe'
& $c scripts/colqwen_encode_pdf_pages.py --prepared data/raw/docling-pages-NEW --reference-contract runs/colqwen-dev-20261107/contract.json --base models/colqwen2-base-range-20261107 --adapter models/colqwen2-v1-20261106 --output runs/colqwen-docling-NEW --device cuda:0
& $p scripts/colqwen_import_pdf_pages.py --prepared data/raw/docling-pages-NEW --encoded runs/colqwen-docling-NEW --index indexes/colqwen-docling-NEW --adapter models/colqwen2-v1-20261106 --report output/colqwen-docling-NEW.json
& $c scripts/colqwen_pdf_score_audit.py --prepared data/raw/docling-pages-NEW --encoded runs/colqwen-docling-NEW --base models/colqwen2-base-range-20261107 --adapter models/colqwen2-v1-20261106 --query 'What are the FIGER and BBN accuracy scores for Ours (Full)?' --page-id 1909.12079__p5 --page-id 1909.12079__p4 --output output/colqwen-pdf-score-NEW.json
& $p -m paper_agent.cli colqwen-hybrid-search 'What are the FIGER and BBN accuracy scores for Ours (Full)?' --index indexes/colqwen-docling-NEW --source-manifest data/raw/docling-pages-NEW/pages.jsonl --image-root data/raw/docling-pages-NEW --colqwen-python $c --worker scripts/colqwen_query_worker.py --base models/colqwen2-base-range-20261107 --adapter models/colqwen2-v1-20261106 --text-index indexes/docling-table-NEW --text-binding data/raw/docling-pages-NEW/binding-table.json --encoder 'models\bge-m3' --clip-index indexes/docling-clip-NEW --clip-python .venv-mm-20261030/Scripts/python.exe --clip-image-model models/multimodal-v1/clip-image-v1 --clip-text-model models/multimodal-v1/clip-text-v1 --evidence-jsonl data/raw/multimodal-eval/docling-v5-agent-reviewed/evidence.jsonl --evidence-root data/raw/multimodal-eval/docling-v5-agent-reviewed/crops > output/colqwen-docling-query-NEW.json
```

## 开发集真实结果与冻结

157 题的开发集逐题计分完成，结果 SHA-256 `818b20286a877f331bcf8459feeadf1eff0010de85c4552530535c6103ec5784`。ColQwen2 只重排每题文本前 50 和全库 CLIP 前 50 的并集，最多 100 页；未使用 gold 或答案生成候选。下表的 ColQwen MRR 对候选外计零；文本、CLIP 是此前全库排序，范围不同，不能解释成同等计算量对比。冲突标签仍为 unknown 并计入分母。

| 开发组／方法 | Hit@1 | Hit@5 | Hit@20 | MRR |
|---|---:|---:|---:|---:|
| 标准 138：文本混合 | .3696 | .7754 | .8986 | .5443 |
| 标准：整页 CLIP | .0072 | .0362 | .0870 | .0237 |
| 标准：候选内 ColQwen2 | .2464 | .5725 | .8696 | .4058 |
| 标准：文本 + ColQwen2 | .3478 | .7826 | .9130 | .5370 |
| 视觉 19：文本混合 | .1053 | .3158 | .4737 | .2122 |
| 视觉：整页 CLIP | .0000 | .0000 | .0526 | .0137 |
| 视觉：候选内 ColQwen2 | .1053 | .3684 | .5263 | .2310 |
| 视觉：文本 + ColQwen2 | .1053 | .3158 | .4211 | .2242 |

候选覆盖标准 131/138、视觉 14/19；未进入候选的目标不可能被候选内模型找回。相对文本，单独 ColQwen 候选排序的标准 Hit@5 改善 7、退化 35；视觉改善 2、退化 1。融合以完整文本页面排名和 ColQwen 前 20 页做 RRF，`k=60`，仅在开发集比较权重 `0/.05/.1/.25/.5/1`，要求两组 Hit@5 不低于文本，再选两组 MRR 之和最高者，冻结权重 `.05`。标准开发题融合 Hit@5 改善 2、退化 1；视觉题均无变化。标准 Hit@1 和视觉 Hit@20 同时下降，不能只报告改善指标。冻结计划 SHA-256 `9f2eaeaafa3fa06c67ea87209a1e892f23a52c4a37652a3d2c3d8d46fb9a390a`，编码／计分冻结文件 SHA-256 `f142939ff44b818c6083c395337de1aed8aa218a16e0bc726f8c6b5b701a5f03`。此前保留组的基线汇总已曝光，因此后续结果不是严格盲测。

开发集逐题重排时间中位数标准约 4.98 秒、视觉约 4.50 秒，页面读取中位数分别约 4.95、4.47 秒，GPU 批量计分中位数约 0.025 秒；这些测量不包含现场模型冷启动及候选生成。所有记录重排时间之和约 784.47 秒，批量与逐页参考的最大差约 `3.88e-6`，采样 RSS 峰值约 1.84 GB，计分 GPU 峰值约 28.6 MB；编码时 GPU 峰值约 4.65 GB。各值来自不同运行，不应相加为单次端到端测量。

保留组独立索引已封存 1,569 页、408,142 token；`sealed.json` SHA-256 `5af6ff9450cb6d88986662f798c4029e8733a95ffc41b0c099d5e22077451d77`，计分后索引目录约 755 MB。其中 1,360 页通过契约与哈希复用开发编码，另 209 页实际重新编码；53 个问题均重新编码。剩余 208 页编码批次约 96.16 秒，GPU 峰值约 4.65 GB，进程结束 RSS 约 1.80 GB；另一次 1 页加 53 问批次约 15.60 秒。索引目录创建到封存约 888 秒，包括两批写入之间的等待、来源校验和并发工作，不能当作纯写入耗时。保留组逐题重排中位数标准 3.63 秒、视觉 4.44 秒；53 题所记录的重排时间之和 212.09 秒，计分 GPU 峰值约 29.4 MB，采样 RSS 峰值约 1.53 GB。这些不是包含加载与候选生成的现场端到端延迟。

## 全部问题和保留组对比

保留组计分在开发计划冻结后执行，结果 SHA-256 `164a446571a7e2d2946865ace8ffba62751ce29b1dfe567e0b12f7bd935f1ba9`。下表单独 ColQwen MRR 将候选外计零；文本、整页 CLIP 和等权文本+CLIP 为全库结果，ColQwen 及其融合只对候选并集做 MaxSim。它们的计算范围与成本不同。视觉 30 题与标准题有论文和页重叠，不是独立测试集。单目标口径下 Hit@K 与该基准的 Recall@K 数值相同。

| 题组／方法 | Hit@1 | Hit@5 | Hit@20 | MRR |
|---|---:|---:|---:|---:|
| 标准 180：BM25 | .4500 | .7778 | .8944 | .5911 |
| 标准：BGE | .4000 | .7333 | .8611 | .5487 |
| 标准：文本混合 | .3944 | .7722 | .9111 | .5606 |
| 标准：整页 CLIP | .0056 | .0278 | .0667 | .0187 |
| 标准：候选内 ColQwen | .2778 | .6222 | .8778 | .4405 |
| 标准：等权文本+CLIP | .0833 | .3000 | .8222 | .2075 |
| 标准：文本+ColQwen .05 | .3778 | .7833 | .9222 | .5558 |
| 视觉 30：BM25 | .1333 | .4000 | .5000 | .2286 |
| 视觉：BGE | .2000 | .3333 | .5000 | .2810 |
| 视觉：文本混合 | .1667 | .4000 | .6000 | .2794 |
| 视觉：整页 CLIP | .0000 | .0000 | .0333 | .0095 |
| 视觉：候选内 ColQwen | .2000 | .4000 | .5333 | .3041 |
| 视觉：等权文本+CLIP | .1000 | .2333 | .5000 | .1915 |
| 视觉：文本+ColQwen .05 | .1667 | .3333 | .5667 | .2833 |

标准全部候选覆盖 171/180，视觉 23/30；保留组分别 40/42、9/11。相对文本，`.05` 融合在标准全部题 Hit@5 改善 4、退化 2，在视觉全部题改善 0、退化 2。标准保留组文本与融合 Hit@5 分别为 .7619、.7857（改善 2、退化 1）；视觉保留组分别为 .5455、.3636（改善 0、退化 2）。这不证明稳定增益；`colqwen-hybrid-search` 默认视觉权重为 0，回退到文本页面排序；仅显式 `--visual-weight .05` 才复现此实验。该实验命令在权重为 0 时仍执行并单独返回 ColQwen 候选结果，因此仍有模型和索引开销；权重 0 不是视觉增益。

为控制候选范围，将已保存的真实 CLIP 全库排序过滤到每题同一候选并集，再与 ColQwen 比较；这不需要重新运行模型。标准全部题同候选 CLIP 的 Hit@1/5/20、MRR 为 `.0056/.0278/.0667/.0288`，ColQwen 为 `.2778/.6222/.8778/.4405`；视觉题分别为 `.0000/.0000/.0333/.0176` 和 `.2000/.4000/.5333/.3041`。候选本身包含 CLIP 前 50，因此前 20 命中与其全库结果相同；这个对照仍保留两者模型、分辨率和计算成本差异。完整开发／保留组及逐题对照在 [同候选对照](colqwen-clip-same-candidates-20261108.json)，SHA-256 `d9f723201b9edf32042d63a4937177733db8258312bf9fc11012040d2813ab54`。

[逐题排名清单](colqwen-page-ranks-20261108.json) 包含 210 个问题 ID、划分、标签状态、候选覆盖和各分支名次，不含问题及转写正文；SHA-256 `c07c08650464437b9a6ef091833826fa8dad6bc3fe3492ed761f673bf212521a`。冲突 `standard-143` 仍为 unknown；目标页空转写 `standard-090` 仍在分母。这里没有把解析错误、标签冲突或检索失败合并成语义正确率，也未测图表 bbox、上传图片或端到端回答。

支线完整回归为 **465 passed、1 skipped**（41.14 秒），包括真实 Chroma 存储、来源守卫、图片与记忆兼容性测试；自动测试不等于语义或人工引用验收。

模型、原数据、索引和结果未提交。完整恢复顺序及构建命令见[恢复记录](IRPAPERS-BOUNDED-RECOVERY-20261107.md)；使用新输出路径，不覆盖任何检查点。候选内排序及融合不得误报为全库 ColQwen2 检索。

下例从本地冻结结果取出 `standard-006` 的**问题文本**，在线阶段只使用问题，随后由文本及 CLIP 产生候选；`.05` 是已冻结的实验权重。索引覆盖不足时命令会报错。`.cache` 用于本地临时文件，输出重定向到新的文件；不需要访问远程生成服务。

```powershell
$p = '.\.venv-gpu\Scripts\python.exe'
$env:TEMP = (Resolve-Path '.cache').Path
$env:TMP = $env:TEMP
$q = & $p -c "import json; print(next(x['question'] for x in map(json.loads,open('runs/irpapers-chroma-baseline-v2-20261106/text-results.jsonl',encoding='utf-8')) if x['case_id']=='standard-006'))"
& $p -m paper_agent.cli colqwen-hybrid-search $q --index indexes/colqwen-dev-20261107 --source-manifest data/raw/irpapers-prepared-20261106/pages.jsonl --image-root data/raw/irpapers-prepared-20261106 --colqwen-python .venv-colqwen-20261106/Scripts/python.exe --worker scripts/colqwen_query_worker.py --base models/colqwen2-base-range-20261107 --adapter models/colqwen2-v1-20261106 --text-index indexes/irpapers-chroma-text-baseline-v2-20261106 --text-binding runs/irpapers-chroma-baseline-v2-20261106/plan.json --encoder models/bge-m3 --clip-index indexes/irpapers-chroma-page-clip-baseline-20261106 --clip-python .venv-mm-20261030/Scripts/python.exe --clip-image-model models/multimodal-v1/clip-image-v1 --clip-text-model models/multimodal-v1/clip-text-v1 --visual-weight .05 > output/colqwen-hybrid-NEW.json
```

根据已保存的两个已完成分区重新计算冻结权重及逐题清单时，先以新文件执行 `scripts/colqwen_fusion_analysis.py choose`（只读开发结果），再用 `evaluate` 分别读取开发和保留结果；保留组还必须提供 `--selection`。`scripts/colqwen_export_ranks.py --help` 给出汇总所需输入；它核验两部分结果、冻结选择与数据哈希，再导出不包含题文的记录。原始二进制、模型和索引均需按[恢复记录](IRPAPERS-BOUNDED-RECOVERY-20261107.md)重建或使用已验证的本地资产。
以下汇总命令本轮以新路径实际重跑，计划、两个融合结果和逐题清单哈希均与首次产物一致；这是对已保存模型结果的汇总复现，不是重新编码或重新执行 210 题模型推理。

```powershell
$p = '.\.venv-gpu\Scripts\python.exe'
$b = 'runs/irpapers-chroma-baseline-v2-20261106'
$d = 'runs/colqwen-dev-20261107/results.jsonl'
$t = 'runs/colqwen-test-20261108/results.jsonl'
$s = 'runs/colqwen-selection-20261108.json'
& $p scripts/colqwen_fusion_analysis.py choose --baseline "$b/text-results.jsonl" --results $d --output runs/colqwen-fusion-plan-NEW.json
& $p scripts/colqwen_fusion_analysis.py evaluate --baseline "$b/text-results.jsonl" --results $d --plan runs/colqwen-fusion-plan-NEW.json --output runs/colqwen-fusion-dev-NEW.json
& $p scripts/colqwen_fusion_analysis.py evaluate --baseline "$b/text-results.jsonl" --results $t --plan runs/colqwen-fusion-plan-NEW.json --selection $s --output runs/colqwen-fusion-test-NEW.json
& $p scripts/colqwen_clip_candidate_control.py --baseline $b --candidates runs/irpapers-colqwen-candidates-20261106 --dev $d --test $t --selection $s --output runs/colqwen-clip-control-NEW.json
& $p scripts/colqwen_export_ranks.py --baseline "$b/text-results.jsonl" --visual "$b/visual-results.jsonl" --dev $d --test $t --fusion-dev runs/colqwen-fusion-dev-NEW.json --fusion-test runs/colqwen-fusion-test-NEW.json --plan runs/colqwen-fusion-plan-NEW.json --selection $s --output runs/colqwen-ranks-NEW.json
```
