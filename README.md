# 科研论文检索 Agent 复现

基于 [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 的工具循环与上下文机制，实现本地论文多轮检索问答：Query 改写、混合检索、重排、证据反思、引用生成和会话记忆。不额外添加 Web UI 或多 Agent。源码来源、固定版本和修改范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[upstream.lock](upstream.lock)。

组件文档总入口：[docs/README.md](docs/README.md)。真实检索评测：[RESULTS.md](reports/qasper-suite/RESULTS.md)。

## 当前可运行内容

- 本地 JSONL / 文本型 PDF 导入，章节与段落切块，BGE tokenizer 滑窗，保留原文字符区间。
- BM25、BGE-M3 + FAISS 精确检索、RRF 和真实 BGE Cross-Encoder 接口。
- 检索评测、逐题预测、数据哈希和索引模型版本记录。
- learn-claude-code 派生的有限工具循环、工具 schema 检查、调用 ID 配对与 usage 汇总。
- 七阶段编排、二次检索上限、引用 ID 校验、session memory 和 trace 的离线集成测试。
- 原始工具结果存档与按 ID 回读；保留近期/尚未消费的结果。

- DeepSeek / OpenAI-compatible Chat Completions 接入，ask/chat、会话恢复、请求上限、结构校验与 usage 记录。
- 批量多轮运行、消融与独立标注评分入口；30×4 来源可追溯的待复核对话草稿。

**验收状态：** 本轮修复评测历史与统计、API 有限重试、归档按需召回和记忆幂等提交。实际验收以 [最终报告](reports/FINAL_REPORT.md)、[真实 API 结果](reports/LIVE_EVAL_RESULTS.md) 为准；剩余问题清单为内部记录，未随仓库发布。QASPER 检索标注与草稿多轮标签分开；未经人工复核的标签、模型 Judge 和离线故障测试不能作为人工 gold。原 data/models/indexes/.env 均直接复用。

**本轮入口：** 原目录运行 `.venv-gpu/Scripts/python.exe -m pytest -q`；完整复现命令见 [评测协议](docs/evaluation.md)。`eval ablation --output 新目录` 会真实执行配置矩阵并消耗 API；`eval compare A B --metric behavior_match` 读取逐例结果做配对簇 Bootstrap。正式数据修订在 `data/eval/v0.3/`，旧版本保留。

## 真实检索结果

QASPER v0.3：100 篇论文、1,996 chunks、100 条官方证据问题；已知论文范围、真实 BAAI 权重、本机 RTX 4060。四组无失败，无付费 API 调用。

| 配置 | Recall@5 | Hit@5 |
|---|---:|---:|
| BM25 | 66.26% | 68.00% |
| BGE-M3 + FAISS | 73.94% | 78.00% |
| 混合检索 + RRF | 74.24% | 77.00% |
| 加 Cross-Encoder 重排 | 77.61% | 81.00% |

这是段落覆盖代理指标，不是答案正确率。提升幅度的置信区间跨零，数据也不是截图中的原始多轮集，不能声称截图数值已复现。完整口径和原始预测见 [评测报告](reports/qasper-suite/RESULTS.md)。

## 本机环境

本机 `agent` 环境实测为 CPU 版；GPU 可用的是 `C:\Users\40533\anaconda3\envs\minimind`。项目 `.venv-gpu` 以其 Python 3.10 创建并继承 PyTorch 2.11.0+cu128，使用 RTX 4060 Laptop 8GB。检索依赖安装在项目虚拟环境；已有 Conda 环境未被修改。

以下命令在项目根目录的 PowerShell 执行：

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
& $py -m pytest -q
```

最新测试结果见 reports/acceptance-tests.xml（历史测试记录单独保留）。交付检查（本地文档链接、保存预测的指标重算、草稿状态）可运行 `& $py scripts/check_artifacts.py`，结果见 [delivery-verification.json](reports/delivery-verification.json)。

其他机器应新建 Python 3.10/3.11 环境，先按 PyTorch 官网安装与显卡驱动匹配的 CUDA 构建，再安装本项目：

```powershell
python -m pip install -e '.[models,api,dev]'
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
& $py -m paper_agent.cli eval retrieval data/qasper/retrieval.jsonl --index indexes/bm25 --mode bm25 --output reports/qasper-bm25-lexical
```

不指定 `--tokenizer` 时采用可重复的词法切分，属于便于安装检查的基线。正式四组对照统一使用 BGE tokenizer，不能把两种 chunk 设置的指标混在一起比较。

## GPU 四组对照（需完整权重）

```powershell
$env:HF_HUB_OFFLINE = '1'
& $py scripts/run_retrieval_suite.py --reuse-index --device cuda:0 --batch-size 4
& $py scripts/report_results.py
```

脚本统一生成 chunks、建立 FAISS IndexFlatIP，再依次评测 BM25、dense、hybrid、rerank。两路各 30 个候选，RRF 常数 60，重排前取 30，返回 5。输出在 `reports/qasper-suite/`。已建好同一版本索引可加 `--reuse-index`；首次必须省略。

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

## 如何理解评测

- `Recall@5` 是前五个 chunk 覆盖的 gold 段落单位比例，先按问题计算再宏平均。相同 gold 段落不重复计分；多标注答案取可替代证据集合中的最大召回。
- 此处为段落覆盖代理指标：长段落的任一窗口命中就计段落命中，不代表完整答案证据均已覆盖。报告会明确这一限制。
- `Hit@5` 只表示是否命中至少一条证据，与 Recall 不同。
- 所有问题都计入分母，失败和缺失预测为零；预测 JSONL 保留错误。
- 引用 ID 校验只证明引用存在及论文匹配，不证明语义蕴含。语义支持和答案质量需要后续人工/模型评测。

## 统一评测入口（eval 子命令）

`python -m paper_agent.cli eval <task>` 提供可复现评测，配置见 [configs/evaluation.json](configs/evaluation.json)，数据集协议与标签来源见 [data/eval/README.md](data/eval/README.md)。

| 子命令 | 内容 | 调 API |
|---|---|---|
| `eval smoke` | 真实 provider 连通 + JSON schema smoke | 是 |
| `eval retrieval <queries>` | BM25/dense/hybrid/rerank 检索指标 | 否 |
| `eval intent <data>` | NLU 意图分类（accuracy/macro-F1/混淆矩阵） | 是 |
| `eval rewrite <data>` | 查询改写实体保留/指代消解诊断 | 是 |
| `eval boundary <data>` | answer/refuse/clarify 决策与拒答 P/R/F1 | 是 |
| `eval memory <data>` | 多轮后记忆检查点一致性 | 是 |
| `eval e2e <data> [--judge]` | 端到端运行 + 可选 LLM-as-Judge（6 维评分） | 是 |
| `eval compare <a> <b>` | 逐例配对差值与 Bootstrap CI | 否 |
| `eval ablation --output <new-dir>` | 执行统一配置的消融矩阵 | 是（LLM 实验） |

LLM-as-Judge 输出 relevance/factual_correctness/completeness/evidence_support/citation_correctness/appropriate_abstention 六维结构化评分与理由，且看不到实验组标签；结果需抽人工复核。真实运行示例与结果见 [reports/LIVE_EVAL_RESULTS.md](reports/LIVE_EVAL_RESULTS.md)。

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

模型名以供应商控制台为准；程序会追加 /chat/completions。请求上限不是金额预算。OpenCode Go 的地址、协议、会话头及编程用途限制见 [API 文档](docs/api.md)，本轮已实测该配置，其他账户/模型需自行验证。

以下命令会使用 API 配额。先单问，再进入同一会话：

```powershell
& $py -m paper_agent.cli ask '请检索论文中与语言模型评测相关的证据。' --index indexes/bge --mode rerank --encoder models/bge-m3 --reranker models/bge-reranker-v2-m3 --device cuda:0 --session research-1
& $py -m paper_agent.cli chat --index indexes/bge --mode rerank --encoder models/bge-m3 --reranker models/bge-reranker-v2-m3 --device cuda:0 --session research-1
```

chat 支持 /corpus、/trace、/memory、/quit。相同 session 恢复事实和近期历史，换 session 隔离。答案附 chunk 引用，完整阶段记录在 runs/traces，会话文件在 runs/memory。密钥与 runs 默认不纳入版本管理。

## 多轮评测与消融

```powershell
& $py scripts/prepare_dialogues.py
# 下列命令使用 API，先确认配额并设置 API_MAX_CALLS。
& $py scripts/run_dialogues.py data/dialogues-draft/dialogues.jsonl --output runs/full --device cuda:0
```

生成 30 组四轮草稿和 100 意图/55 边界/70 端到端视图，需要独立复核。默认 100 次 API 上限不足以跑完所有轮，须按预算调整；失败预测不会从分母移除。输出目录必须是新目录。

审阅 judgments-template.jsonl，另存 reviewed.jsonl，再运行：

```powershell
& $py scripts/score_dialogues.py runs/full data/reviewed.jsonl --view data/dialogues-draft/e2e-view.jsonl --output runs/full/e2e-metrics.json
```

支持 --no-rewrite、--no-reflect、--no-compact 及四种 --mode；每次只改一个因素，用新输出目录。口径、审阅要求、记忆实验限制见 [评测与复现](docs/evaluation.md)。

## 组件说明

| 组件 | 文档 | 实现 |
|---|---|---|
| 七阶段架构 | [架构](docs/architecture.md) | pipeline.py |
| 数据导入/分块 | [语料](docs/corpus.md) | corpus.py |
| RAG 与混合检索 | [检索](docs/rag-retrieval.md) | retrieval.py |
| Cross-Encoder | [重排](docs/reranking.md) | retrieval.py |
| 意图/指代/改写 | [路由](docs/routing.md) | providers.py |
| 证据检查与拒答 | [反思](docs/reflection.md) | pipeline.py |
| 事实与引用 | [生成](docs/generation.md) | memory.py |
| 会话与压缩 | [记忆](docs/memory.md) | memory.py / pipeline.py |
| 工具循环/trace | [追踪](docs/runtime-tracing.md) | runtime.py / pipeline.py |
| API 适配 | [API](docs/api.md) | providers.py |
| 运行/标注/评分 | [评测](docs/evaluation.md) | scripts/、task_metrics.py |

复现计划、上游复用审查与 references/ 快照为内部文档，未随仓库发布；当前执行状态见 [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md)。


## 许可

本项目自有代码采用 [MIT License](LICENSE)。

- 上游 [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 的移植部分（`src/paper_agent/runtime.py`、`src/paper_agent/memory.py`、`tests/test_upstream_compact_portable.py`）保留其 MIT 声明，全文见 [licenses/learn-claude-code-LICENSE](licenses/learn-claude-code-LICENSE)。
- BGE-M3 / bge-reranker-v2-m3 权重与 QASPER v0.3 数据不随本仓库发布，遵循各自的模型卡与数据许可；来源、固定版本与许可记录见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- `models/`、`indexes/`、`runs/`、`data/qasper/`、`references/` 与 `.env` 均已排除，不随仓库分发。

本轮实际 API 数据冻结为 v0.3；后续 v0.4 补充显式 history session/turn 元数据、论文 ID/标题别名和实体指标适用范围。v0.4 的 Rewrite 分数是对保存的 v0.3 预测离线重算（`reports/acceptance/rewrite/rescored-v04.json`），不是新 API 实验；两版均保留草稿标签与原始结果。
