# 论文图文检索：基础阶段

本功能使用独立环境运行 Docling 和 CLIP；既有文本环境、索引和会话不需要迁移。五篇论文的真实运行、agent 逐例复核及剩余问题见 [本轮复核报告](../reports/revision/MULTIMODAL-AGENT-REVIEW-20261031.md)；原始首轮结果保留在 [此前阶段报告](../reports/revision/MULTIMODAL-2026-10-31.md)。用户已授权 agent 完成原定复核；agent 复核不是人类金标。

Docling 从 PDF 页面渲染图表裁剪，而不是只抽取嵌入图像。每个条目使用 `(paper_id, chunk_id)`；图表保存页码、页面坐标、旋转、编号（无法确认时为空）、图注、PDF 和图片哈希及解析版本。当前旋转页会明确拒绝处理；坐标和哈希校验只能证明记录内部一致，仍须逐页检查裁剪与原论文的对应关系。

图像模型固定为 `sentence-transformers/clip-ViT-B-32`，文本模型固定为 `sentence-transformers/clip-ViT-B-32-multilingual-v1`，参照[多语言模型卡](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)。两者的 commit 分别为 `327ab6726d33c0e22f920c83f2ff9e4bd38ca37f` 和 `58edf8cada9e398793dca955574a48cbb7f18be2`。构建时记录快照文件哈希、维度和索引哈希；编码在独立进程中执行，图像与文本模型分时加载。向量均归一化，视觉 FAISS 只存 CLIP 图像向量，不与 BGE 向量计算相似度。

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

本次纯 CLIP 检索的命令如下；输出路径必须使用新文件：

```powershell
& .\.venv-gpu\Scripts\python.exe scripts/run_clip_retrieval_eval.py --visual-python .\.venv-mm-20261030\Scripts\python.exe --image-model models/multimodal-v1/clip-image-v1 --text-model models/multimodal-v1/clip-text-v1 --visual-index indexes/multimodal-v5-agent-reviewed --image-root data/raw/multimodal-eval/docling-v5-agent-reviewed/crops --output output/clip-retrieval-new.json --device cuda:0
```

问答首轮配置见本地 `runs/multimodal-acceptance-20261031/config.json`，本轮针对性复验配置见 `runs/multimodal-review-fixes-20261031/config.json`。60 次总额度已用完；如需未来继续真实模型测试，必须取得新的明确授权并记录新预算，不能删除标记或创建新账本规避现有上限。

图文问答可选择 `--visual-claim-check`：生成且引用合法后，同一模型仅依据被引证据和相应图片逐项检查；任何不支持、未知、缺失或失败导致受控拒答。该选项增加一次计费请求，默认关闭，检查结果不是独立语义证明。关联正文只有在论文、chunk、PDF 哈希、页码、原文及显式图号引用均匹配可信文本索引时，才作为独立证据加入。

### 图注、表格语义索引和保留集

需要先按上述步骤准备正文密集索引及独立 CLIP 索引，然后在新的输出目录建立独立 BGE 图注和表格文本索引；不要覆盖已有目录：

```powershell
& .\.venv-gpu\Scripts\python.exe -m paper_agent.cli visual-semantic-index data/raw/multimodal-holdout-v1-docling/evidence.jsonl --visual-root data/raw/multimodal-holdout-v1-docling/crops --output indexes/multimodal-holdout-v1-semantic --encoder models/bge-m3 --device cpu --batch-size 4 --max-length 768
```

普通问答可在既有视觉参数之外指定 `--visual-semantic-index indexes/multimodal-holdout-v1-semantic`、`--mode rerank --encoder models/bge-m3` 和相同 `--max-length 768`；密集正文索引、BGE 图注索引和 CLIP 索引必须由同一版证据生成，编码模型路径及快照必须匹配元数据。此索引仍可能缺少表号，按编号过滤前应检查来源，不能用图注名称猜测表格区域。

本次保留集的原始配置和输出在 `runs/multimodal-holdout-v1-eval-20261031/`，逐次请求在 `requests/`，完整调用参数可由 `scripts/run_multimodal_holdout.py --help` 查看。冻结题集、独立原页复核、实现哈希、单题结果及局限见 `reports/revision/MULTIMODAL-HOLDOUT-20261031.md`。模型调用没有自动重试；中断后要先核查本地的 `reserved_unknown` 及 provider 用量，不能将未知状态当作未消费请求。新请求需要已有授权，不能通过新建账本规避原有限额；本次新账本和旧 60 次账本分别保留。

## 后续 ColQwen2

后续可单独添加页面级多向量编码器、带版本和页面映射的独立索引，以及文本查询的 MaxSim 排序，再与现有图表级 CLIP 排名融合；图片搜图继续由 CLIP 完成。应先测单页的显存、编码耗时、向量与磁盘规模，再以固定页面标注检验召回、引用坐标和中文问题；8 GB 显存可能需要低批量、量化或 CPU，具体能力和成本尚未实测。该阶段不属于本次基础闭环的完成条件。
