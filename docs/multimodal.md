# 论文图文检索：基础阶段

## 图片到文本与图片问答

CLIP 文本索引覆盖同一文本索引中具有有效页面坐标的正文和图注；新建索引默认使用独立 Chroma collection，早前评测使用的 FAISS 索引保留为可选兼容。它与 BGE 索引独立，图像模型与多语言文本模型使用下文固定的两个对齐快照；不能将两种模型的向量混算。每个 CLIP 表示保存原始 chunk ID、字符偏移与精确切片；使用本地 fast tokenizer 的 128 token 上限，预留 2 个特殊 token、126 token 窗口和 16 token 重叠。长 chunk 不改写 BGE 索引。解析输出和图注关联仍未自动成为人工核实的原文；没有可靠可定位来源的表格提取不作为已验证表格文本加入。

下面使用本机已有双论文开发索引，输出目录必须为新目录。命令需在项目根目录执行，两个模型路径必须包含原有完整性标记；主环境需安装 `pip install -e '.[visual]'` 对应的 Pillow 和项目依赖，独立模型环境使用已有 Sentence Transformers。必要时把进程临时目录设为工作区 `.cache`，不会修改 `.env`：

```powershell
$env:TEMP = (Resolve-Path .cache).Path
$env:TMP = $env:TEMP
$py = '.\.venv-gpu\Scripts\python.exe'
$vpy = '.\.venv-mm-20261030\Scripts\python.exe'
$visual = @('--visual-index','indexes/chroma-migration-clip-20261106', '--visual-root','data/raw/multimodal-final-dev-20261102/crops', '--visual-python',$vpy, '--visual-image-model','models/multimodal-v1/clip-image-v1', '--visual-text-model','models/multimodal-v1/clip-text-v1', '--visual-device','cuda:0')
& $py -m paper_agent.cli clip-text-index indexes/chroma-migration-text-20261106/chunks.jsonl @visual --output indexes/clip-text-new
& $py -m paper_agent.cli clip-search-image-text @visual --clip-text-index indexes/clip-text-new data/raw/multimodal-holdout-v3-20261101/v3img01.png
& $py -m paper_agent.cli clip-search-image-text @visual --clip-text-index indexes/clip-text-new --kind caption data/raw/multimodal-holdout-v3-20261101/v3img01.png
& $py -m paper_agent.cli visual-search-text @visual --caption-fusion 'three stages of ULMFiT'
& $py -m paper_agent.cli ask --index indexes/chroma-migration-text-20261106 --mode bm25 @visual --clip-text-index indexes/clip-text-new --image data/raw/multimodal-holdout-v3-20261101/v3img01.png --session image-example 'What stages are shown in this image?'
& $py -m paper_agent.cli ask --index indexes/chroma-migration-text-20261106 --mode bm25 @visual --clip-text-index indexes/clip-text-new --image data/raw/multimodal-holdout-v3-20261101/v3img01.png --session image-only-example
```

`--image` 可重复两次；`ask` 的文本位置参数可省略，`chat` 启动时提供的图片在该进程的每轮使用。旧文本命令不需要 CLIP 文本索引。图片输入必须是本地 PNG/JPEG，最多 20 MiB、4000 万像素；图片只作查询输入，不能作为论文证据引用。每次图片编码同时用于图搜图和图搜正文；图片分支与文本分支保留各自排名，按排名交错并去重，不把不同实体说成共同命中。图、独立图注和正文各自保留原始 ID、页码和框；仅有唯一同论文、同 PDF、同页且文字一致的图注可作为候选扩展，关系仍标未核实。模型与数据指纹、索引校验和、数量或维度不符会报错；图片检索单分支失败或无结果的状态写入本地 trace。

Reflect 和 Generate 每次调用最多接收三张图片：用户图占用其中名额，其余可用于经校验的论文图。用户图片和论文裁图会在真实请求中发给 `.env` 配置的远程视觉模型；trace 和请求账本不保存原图或 base64，用户原文件不会被删除。请勿依据 CLIP 相似度断定细小数字、箭头或语义；必须核对实际出处与引用。此处仅进行小样本、已见题目的开发评测，直接图片到指定图注的 Hit@5 为 0/3；三条图搜图命中后经位置关联扩展找到图注为 3/3，不能算成直接检索命中。混合问答案例及纯图片回答的相关性缺陷、逐例结果和复现命令见 [本轮报告](../reports/revision/MULTIMODAL-IMAGE-TEXT-20261106.md)。

后续修复增加可选 `--caption-fusion`，只对同一论文图像实体融合 CLIP 和图注词项排名；`--kind caption` 则只在独立文本索引的图注子集中直接检索，不使用图搜图关联。两个选项可能改善特定查询，但不保证跨语言召回。纯图片查询需要每张图在图搜图中有至少两个候选，最高相似度至少 0.90 且领先第二名至少 0.05；否则直接返回 `image_match_uncertain`。通过门槛后，仅把最高候选及其独立关联证据交给 Reflect，模型仍须检查图像，分数本身不是身份或语义证明。这是小样本启发式，可能拒绝有效图片；最多两张用户图还受每次模型请求的总图片数约束。可选 BGE 图注或表格分支查询失败时会记录 `failed` 并保留其他分支，构建或加载阶段的索引不一致仍明确报错。[后续修复和冻结论文验证](../reports/revision/MULTIMODAL-IMAGE-RETRIEVAL-REPAIR-20261106.md)单独记录，不改写上述早期成绩。

本功能使用独立环境运行 Docling 和 CLIP；既有文本环境、索引和会话不需要迁移。五篇论文的真实运行、agent 逐例复核及剩余问题见 [本轮复核报告](../reports/revision/MULTIMODAL-AGENT-REVIEW-20261031.md)；原始首轮结果保留在 [此前阶段报告](../reports/revision/MULTIMODAL-2026-10-31.md)。用户已授权 agent 完成原定复核；agent 复核不是人类金标。

Docling 从 PDF 页面渲染图表裁剪，而不是只抽取嵌入图像。每个条目使用 `(paper_id, chunk_id)`；图表保存页码、页面坐标、旋转、编号（无法确认时为空）、图注、PDF 和图片哈希及解析版本。当前旋转页会明确拒绝处理；坐标和哈希校验只能证明记录内部一致，仍须逐页检查裁剪与原论文的对应关系。

图像模型固定为 `sentence-transformers/clip-ViT-B-32`，文本模型固定为 `sentence-transformers/clip-ViT-B-32-multilingual-v1`，参照[多语言模型卡](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)。两者的 commit 分别为 `327ab6726d33c0e22f920c83f2ff9e4bd38ca37f` 和 `58edf8cada9e398793dca955574a48cbb7f18be2`。构建时记录快照文件哈希、维度和向量指纹；编码在独立进程中执行，图像与文本模型分时加载。向量均归一化，视觉 Chroma collection 只存 CLIP 图像向量，不与 BGE 向量计算相似度；余弦距离转换为 `1-distance` 才是相似度。历史 FAISS 为精确搜索，Chroma HNSW 是近似搜索。

当 Chroma 0.6.3 在全量候选查询时返回已识别的 HNSW 连续数组错误，适配器从该 collection 读取向量并执行精确余弦全扫描，策略记录为 `chroma_exact_full_scan_after_hnsw_limit`；这可能增加内存和耗时，普通较小候选查询仍使用 HNSW。整页 IRPAPERS 基线、回退的实际使用、失败记录和复现命令见 [运行记录](../reports/revision/CHROMA-IRPAPERS-20261106.md)。整页 CLIP 的结果不等于图表裁剪检索或引用验证。

本次实际环境为 `.venv-mm-20261030`：Python 3.10.11、Docling 2.48.0、Sentence Transformers 3.4.1、FAISS CPU 1.9.0.post1、PyTorch 2.6.0+cu124。原有缓存写入卡住，通过 `--no-cache-dir` 和官方 HTTPS 包源解决。准备全新环境时，目录必须不存在，不要覆盖现有环境。下列为 CPU 安装示例；GPU 需另行安装适合本机的官方 PyTorch CUDA wheel，复核版本及显存：

```powershell
& .\.venv\Scripts\python.exe -m venv .venv-visual-new
& .\.venv-visual-new\Scripts\python.exe -m pip install --no-cache-dir 'docling==2.48.0' 'sentence-transformers==3.4.1' 'faiss-cpu==1.9.0.post1' 'pypdf==5.9.0'
```

若使用此新目录，将下面示例中的 `$vpy` 改为 `.\.venv-visual-new\Scripts\python.exe`。不要为绕过网络错误而关闭证书校验；在新的隔离目录中安装并核实实际版本。

在准备好的隔离环境中，以全新目录下载两个快照（不要覆盖已有模型）：

```powershell
$vpy = '.\.venv-mm-20261030\Scripts\python.exe'
& $vpy scripts/clip_worker.py download sentence-transformers/clip-ViT-B-32 327ab6726d33c0e22f920c83f2ff9e4bd38ca37f models/visual-clip-image-v1
& $vpy scripts/clip_worker.py download sentence-transformers/clip-ViT-B-32-multilingual-v1 58edf8cada9e398793dca955574a48cbb7f18be2 models/visual-clip-text-v1
```

对每篇论文使用不同输出，显式指定与文本索引一致的论文 ID；源 PDF 和裁剪图均留在本地、不提交到公开仓库。以下示例需要替换本地路径：

解析输出另生成 `.quarantine.jsonl` 与 `.summary.json`：零面积、非有限、反向、越界或来源缺失的局部条目被隔离并保留源标识和原因，不生成可引用区域；空文本单独标为跳过，可能包括公式，须检查遗漏。缺页和不完整转换仍报错。输出和旁车均须使用新路径。BART 的受控图表复核及逐页开发复验见 `reports/revision/MULTIMODAL-PARSE-REPAIR-20261102.md`；仅对显式原页复核、PDF 哈希和原始坐标均匹配的对象补关联，不把它推广为自动多面板识别。

双论文上一版的索引、模型请求账本和 9+3 开发复验见 `reports/revision/MULTIMODAL-FINAL-DEVELOPMENT-20261102.md`；该报告保留当时首页候选漏召回及 h07 误拒的原始结果。后续版本支持明确的普通论文 ID 标题／作者提问，并检查真实首页源块；不可靠或未知范围应澄清，来源标题元数据不代替正文。表号引用可使用同 PDF、同页、身份一致且有明确来源关联的独立图注，仍须核对语义。`this effort` 的有界改写必须引用可定位原文并经声明检查；集合规模或语义条件缺失时仍会拒答，不能将该机制视为任意指代均可恢复。新旧账本独立保留，下文旧验收运行器的 60 次标记和历史结果不得覆盖或回写。

本轮修复、真实开发复验、独立 agent 原页复核与未解决的 v305/v309 引用限制见 `reports/revision/MULTIMODAL-ACCEPTANCE-CLOSEOUT-20261102.md`。集合规模只在问题、已检索原文、单一论文及来源位置匹配时进行有界补全，并须逐声明自检；无法可靠匹配时保持拒答。旧声明重放和新生成结果分开记账。

```powershell
& $vpy scripts/docling_ingest.py data/raw/example.pdf --paper-id example --title 'Example paper' --image-root data/raw/visual-v1/crops --output data/raw/visual-v1/example.jsonl
$py = '.\.venv\Scripts\python.exe'
& $py -m paper_agent.cli visual-collect data/raw/visual-v1/example.jsonl --evidence-output data/raw/visual-v1/evidence.jsonl --text-output data/raw/visual-v1/text.jsonl
& $py -m paper_agent.cli index data/raw/visual-v1/text.jsonl --output indexes/visual-text-v1
$visual = @('--visual-index','indexes/visual-v1','--visual-root','data/raw/visual-v1/crops','--visual-python',$vpy,'--visual-image-model','models/visual-clip-image-v1','--visual-text-model','models/visual-clip-text-v1','--visual-device','cpu')
& $py -m paper_agent.cli visual-index @visual data/raw/visual-v1/evidence.jsonl
& $py -m paper_agent.cli visual-search-text @visual --paper-id example --figure-number 'Figure 1' 'architecture diagram'
& $py -m paper_agent.cli visual-search-image @visual data/raw/query.png
& $py -m paper_agent.cli ask --index indexes/visual-text-v1 --mode bm25 @visual --session visual-demo 'Describe Figure 1 in example.'
```

`visual-collect` 也接受多个逐篇 JSONL；两个输出必须是新文件。文本索引可按原有方式使用 BGE-M3 和重排器。视觉查询支持论文与 Figure/Table 编号过滤；图注、文本和视觉候选按实体排序融合，文本重排器只处理文本。图文问答要求开启 Reflect，最多向模型发送三张经路径、页码、坐标与哈希检查的本地图。模型只能引用检索结果；页码、坐标和路径由本地记录回填。引用格式合法不等于结论受证据支持。

图片数据仅在模型请求时编码，不存入日志、Trace 或记忆；可选 Langfuse 投影只发白名单元数据，不发送论文正文或图片。真实试验必须另设请求总数、超时、输出 token 和零重试上限，并保留 provider usage；普通命令的默认预算由原项目配置决定，不应误认为自动满足特定试验额度。

本机真实验收使用 `models/multimodal-v1/clip-image-v1` 和 `models/multimodal-v1/clip-text-v1`，Docling 权重使用 `models/docling-mm-20261030-v2`。每篇导入时可传 `--device cuda:0 --artifacts-path models/docling-mm-20261030-v2 --trace-dir runs/multimodal-parse`。固定题集的原始逐篇导出在 `data/raw/multimodal-eval/docling-v2/`，问答所用的修订证据在 `data/raw/multimodal-eval/docling-v3-reviewed/`，索引为 `indexes/multimodal-text-bge-20261030` 和 `indexes/multimodal-v3-reviewed`。原始与修订数据均保留；详见报告，不可将人工确认的特定区域修订当作通用自动解析能力。

之后排除另一个已确认误关联形成 v4；本轮依据原页重建完整三面板图并去除重复图块形成 `data/raw/multimodal-eval/docling-v5-agent-reviewed/` 和 `indexes/multimodal-v5-agent-reviewed`。纯 CLIP 检索与针对性问答已在 v5 真实运行，原首轮问答使用 v3，早期定向复验使用 v4。新的使用优先选 v5，并继续检查其他未知误关联。

验收运行器 `scripts/run_multimodal_acceptance.py` 对该任务使用持久预算文件；本阶段已产生 **60 次调用**，额度已耗尽，不得删除该文件或使用新账本绕过上限。固定案例的纯 CLIP 评测命令使用 `scripts/run_clip_retrieval_eval.py`，逐例记录器使用 `scripts/multimodal_eval_record.py`。agent 已代做复核，仍不能表述为人类审查。

原五篇论文和 13 个固定案例的首轮状态、后续开发复验、实际用量及当前限制汇总于 [第一阶段状态核对](../reports/revision/MULTIMODAL-PHASE1-STATUS-20261107.md)。首轮问答使用 v3，v5 是之后的修订版本；复现或比较时应分别记录版本，且原阶段请求预算已耗尽。

本次纯 CLIP 检索的命令如下；输出路径必须使用新文件：

```powershell
& .\.venv-gpu\Scripts\python.exe scripts/run_clip_retrieval_eval.py --visual-python .\.venv-mm-20261030\Scripts\python.exe --image-model models/multimodal-v1/clip-image-v1 --text-model models/multimodal-v1/clip-text-v1 --visual-index indexes/multimodal-v5-agent-reviewed --image-root data/raw/multimodal-eval/docling-v5-agent-reviewed/crops --output output/clip-retrieval-new.json --device cuda:0
```

问答首轮配置见本地 `runs/multimodal-acceptance-20261031/config.json`，本轮针对性复验配置见 `runs/multimodal-review-fixes-20261031/config.json`。60 次总额度已用完；如需未来继续真实模型测试，必须取得新的明确授权并记录新预算，不能删除标记或创建新账本规避现有上限。

图文问答可选择 `--visual-claim-check`：生成且引用合法后，同一模型仅依据被引证据和相应图片逐项检查，默认关闭。若同时用 `--local-claim-audit` 指定全新的受控本地目录，当部分声明有争议时可对争议声明做一次复核；原声明和两次 verdict 保留，冲突标 unknown 并受控拒答，不静默删掉可能必要的声明。文本数字或单位缺失时最多一次显式引用修复，图像中的数字不能由文本规则自动证明；明显的跨声明指代或缺少可核查 test 数据集限定也会受控拒答。这些步骤会增加计费请求，同模型自检和数字重合都不是独立语义证明。图形、图注和关联正文分别引用；论文标题作为本地索引来源元数据单列且标注未独立验证，不是局部框内原文。

### 图注、表格语义索引和保留集

需要先按上述步骤准备正文密集索引及独立 CLIP 索引，然后在新的输出目录建立独立 BGE 图注和表格文本索引；不要覆盖已有目录：

```powershell
& .\.venv-gpu\Scripts\python.exe -m paper_agent.cli visual-semantic-index data/raw/multimodal-holdout-v1-docling/evidence.jsonl --visual-root data/raw/multimodal-holdout-v1-docling/crops --output indexes/multimodal-holdout-v1-semantic --encoder models/bge-m3 --device cpu --batch-size 4 --max-length 768
```

普通问答可在既有视觉参数之外指定 `--visual-semantic-index indexes/multimodal-holdout-v1-semantic`、`--mode rerank --encoder models/bge-m3` 和相同 `--max-length 768`；密集正文索引、BGE 图注索引和 CLIP 索引必须由同一版证据生成，编码模型路径及快照必须匹配元数据。此索引仍可能缺少表号，按编号过滤前应检查来源，不能用图注名称猜测表格区域。

本次保留集的原始配置和输出在 `runs/multimodal-holdout-v1-eval-20261031/`，逐次请求在 `requests/`，完整调用参数可由 `scripts/run_multimodal_holdout.py --help` 查看。冻结题集、独立原页复核、实现哈希、单题结果及局限见 `reports/revision/MULTIMODAL-HOLDOUT-20261031.md`。模型调用没有自动重试；中断后要先核查本地的 `reserved_unknown` 及 provider 用量，不能将未知状态当作未消费请求。新请求需要已有授权，不能通过新建账本规避原有限额；本次新账本和旧 60 次账本分别保留。

第二版保留集及本轮证据修复见 `reports/revision/MULTIMODAL-V3-20261031.md`。`scripts/run_multimodal_v3_eval.py --help` 列出真实评测参数，`--decompose` 启用带范围保护的分论文检索；题集必须与指定 SHA-256 一致，输出目录必须不存在。请求状态写入 `requests/`，原始声明与检查写入 `claim-audit/`；`scripts/join_multimodal_claim_audit.py` 可只读连接原始和修复后引用的本地坐标，输出新文件。不要把不包含正文的审计汇总当作语义证明。原 PDF、图片、模型、索引和运行结果均只在本机，复现前须依据题集说明从原始来源取得并校验。

2026-11-01 收尾结果和源码冻结见 `reports/revision/MULTIMODAL-V4-20261101.md`。BERT 两幅复合图基于经 agent 核查的 `data/eval/multimodal-holdout-v2/agent-reviewed-composites-v4.json`，可使用 `scripts/apply_reviewed_composites.py --help` 的 `--panel-manifest-output` 保留来源区域；这不是任意多面板的自动正确解析。新保留集一次冻结后 ULMFiT 因零宽 bbox 中止，BART 的单论文部分诊断另行标注，不能作为原定两论文完整结果。原始 PDF、旧索引和请求账本不可覆盖；如后续修复零宽 bbox，应先补上防误定位测试，保留失败结果并按是否使用该集调参重新确定评测边界。

## 后续 ColQwen2

后续可单独添加页面级多向量编码器、带版本和页面映射的独立索引，以及文本查询的 MaxSim 排序，再与现有图表级 CLIP 排名融合；图片搜图继续由 CLIP 完成。应先测单页的显存、编码耗时、向量与磁盘规模，再以固定页面标注检验召回、引用坐标和中文问题；8 GB 显存可能需要低批量、量化或 CPU，具体能力和成本尚未实测。该阶段不属于本次基础闭环的完成条件。

2026-11-06 的实验增加了独立的页面 token Chroma 存储和候选内 MaxSim 算法，以及去重页面的可选排名融合；原始图片检索和引用默认路径未更改。当时基础模型权重下载受阻，尚未产生真实模型向量或基准成绩。候选覆盖、当时的依赖问题以及文本方案在视觉题上的退化见 [召回诊断与运行报告](../reports/revision/IRPAPERS-RECALL-COLQWEN-20261106.md)。

2026-11-07 的后续工作已恢复兼容依赖和固定权重，并实际运行两页 ColQwen2 GPU 前向、Chroma 写入及候选内 MaxSim；完整 210 题评测尚未完成。独立有界 Chroma 查询、可恢复的多向量页面存储、页面切片开发探针、下载断点受损情况及恢复步骤见 [本轮记录](../reports/revision/IRPAPERS-BOUNDED-RECOVERY-20261107.md)。有界搜索及切片结果尚未替换默认问答路径。
