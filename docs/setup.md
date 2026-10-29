# 安装、资产准备与运行

已有本机资产直接复用；新机器按以下步骤准备。完整验收结果见 [最终报告](../reports/FINAL_ACCEPTANCE.md)。

## 本机环境

本机 `agent` 环境实测为 CPU 版；GPU 可用的是 `C:\Users\40533\anaconda3\envs\minimind`。项目 `.venv-gpu` 以其 Python 3.10 创建并继承 PyTorch 2.11.0+cu128，使用 RTX 4060 Laptop 8GB。检索依赖安装在项目虚拟环境；已有 Conda 环境未被修改。

以下命令在项目根目录的 PowerShell 执行：

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
& $py -m pytest -q
```

修订后的测试结果见 reports/revision/FINAL.md；reports/final-acceptance/final-v2-tests.xml 是修订前测试。`& $py scripts/check_artifacts.py` 只读检查历史评测数据及文档链接，并报告当前源码相对历史冻结的变化，不覆盖冻结报告。

其他机器应新建 Python 3.10/3.11 环境，先按 PyTorch 官网安装与显卡驱动匹配的 CUDA 构建，再安装本项目：

```powershell
python -m pip install -e '.[models,api,memory,dev]'
```

模型首次下载需要联网，但没有 Claude/API 费用。下载脚本只获取 BAAI 官方模型并记录不可变 revision：

```powershell
$env:HF_HOME = (Join-Path (Get-Location) '.cache/huggingface')
& $py scripts/download_models.py --model BAAI/bge-m3
& $py scripts/download_models.py --model BAAI/bge-reranker-v2-m3
```

脚本内冻结了两个模型的 revision。若 Hugging Face 大文件下载停滞，可加 `--source modelscope`，从 BAAI 的另一发布渠道分段续传；仍须匹配固定 Hugging Face 权重 SHA256 才会启用，不接受哈希不一致的文件。


## 数据准备

```powershell
& $py scripts/prepare_qasper.py --papers 100 --queries 100
```

数据来自官方 QASPER v0.3 test 发布包（CC BY 4.0），使用 seed=42 固定抽样。问题保留官方标注，仅保留能唯一映射到全文段落的文本证据。官方测试集只用于报告基线，不能用它优化参数后声称未见测试。

这 100 条是英文单轮检索题，不是截图的中文多轮测试集。QASPER 的问题以指定论文为上下文，因此主评测保留已知 paper_id 范围；另存 retrieval-global.jsonl 做无论文提示的全库压力测试，两种任务必须分开报告。论文全文被索引，答案标注不会被索引。不可回答与图表证据样例未进入本次检索指标；这不意味着系统具备拒答或图表能力。数据 manifest 记录抽样和排除统计。


## 快速运行 BM25（无需模型权重）

```powershell
& $py -m paper_agent.cli ingest data/qasper/papers.jsonl --output data/qasper/chunks-lexical.jsonl
& $py -m paper_agent.cli index data/qasper/chunks-lexical.jsonl --output indexes/bm25
& $py -m paper_agent.cli search 'What retrieval method is used?' --index indexes/bm25 --mode bm25
& $py -m paper_agent.cli eval retrieval data/qasper/retrieval.jsonl --index indexes/bm25 --mode bm25 --output reports/reproduced-bm25
```

不指定 `--tokenizer` 时采用可重复的词法切分，属于便于安装检查的基线。正式四组对照统一使用 BGE tokenizer，不能把两种 chunk 设置的指标混在一起比较。


## GPU 四组对照（需完整权重）

```powershell
$env:HF_HUB_OFFLINE = '1'
& $py scripts/run_retrieval_suite.py --reuse-index --device cuda:0 --batch-size 4 --output reports/reproduced-retrieval
& $py scripts/report_results.py --run reports/reproduced-retrieval
```

脚本统一生成 chunks、建立 FAISS IndexFlatIP，再依次评测 BM25、dense、hybrid、rerank。两路各 30 个候选，RRF 常数 60，重排前取 30，返回 5。输出在 `reports/reproduced-retrieval/`。已建好同一版本索引可加 `--reuse-index`；首次必须省略。

```powershell
& $py -m paper_agent.cli search 'How is the model evaluated?' --index indexes/bge --mode rerank --encoder models/bge-m3 --reranker models/bge-reranker-v2-m3 --device cuda:0
& $py -m paper_agent.cli shell --index indexes/bm25 --mode bm25
```

离线 shell 支持 `/corpus`、`/trace`、`/memory`、`/quit`。其中 `/memory` 明确显示离线检索不生成事实记忆；论文问答的记忆更新由 `PaperPipeline` 管理，不能用此 shell 冒充完整问答 Agent。


## 导入自己的语料

JSONL 每行包含 `paper_id`、`title`、`source_url` 和 `paragraphs`，段落字段为 `paragraph_id`、`section`、`text`、可选 `page`。具体结构可查看 `data/qasper/papers.jsonl`。

```powershell
& $py -m paper_agent.cli pdf path/to/paper.pdf --output data/my-paper.jsonl
& $py -m paper_agent.cli ingest data/my-paper.jsonl --output data/my-chunks.jsonl --tokenizer models/bge-m3
```

PDF 双栏阅读顺序、章节、表格需人工检查；没有可靠页码的结构化全文以章节/段落引用，不能补造页码。扫描件不做 OCR。


## 配置 API 与多轮问答

本轮固定复用已有 OpenCode .env。以下模板仅用于全新环境，已有 .env 不复制覆盖：

```powershell
Copy-Item .env.example .env
```

```dotenv
API_KEY=你的密钥
API_BASE_URL=https://opencode.ai/zen/go/v1
MODEL_ID=deepseek-v4.1-flash
API_JSON_MODE=true
API_MAX_CALLS=100
```

模型名以供应商控制台为准；程序会追加 /chat/completions。请求上限不是金额预算。OpenCode Go 的地址、协议、会话头及编程用途限制见 [API 文档](../docs/api.md)，本轮已实测该配置，其他账户/模型需自行验证。

以下命令会使用 API 配额。先单问，再进入同一会话：

```powershell
& $py -m paper_agent.cli ask '请检索论文中与语言模型评测相关的证据。' --index indexes/bge --mode rerank --encoder models/bge-m3 --reranker models/bge-reranker-v2-m3 --device cuda:0 --session research-1
& $py -m paper_agent.cli chat --index indexes/bge --mode rerank --encoder models/bge-m3 --reranker models/bge-reranker-v2-m3 --device cuda:0 --session research-1
```

chat 支持 /corpus、/trace、/memory、/quit。相同 session 恢复近期历史和已确认记忆；跨会话共享须显式配置项目范围。答案附 chunk 引用，完整阶段记录在 runs/traces，会话数据在 runs/memory。密钥与 runs 默认不纳入版本管理。


## 完整三层问答

默认 ask/chat 使用 SQLite 本地模式。要启用 Redis 与 Chroma 投影，启动 Redis 并设置：

```powershell
$env:MEMORY_BACKEND='three_layer'
$env:REDIS_URL='redis://localhost:16379/0'
& $py -m paper_agent.cli chat --index indexes/bge --mode rerank --encoder models/bge-m3 --reranker models/bge-reranker-v2-m3 --device cuda:0 --session research-1
```

Redis 地址按本机实际服务修改。三层模式需要 encoder，不能用不加载 encoder 的 BM25 CLI 代替。
