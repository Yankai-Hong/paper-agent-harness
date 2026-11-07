# Docling 与 CLIP 页面检索：真实 PDF 小样本（2026-11-08）

本次在原有五篇已解析 PDF 上建立页面检索，目的是验证来源和调用链，不将其当作 IRPAPERS 或独立测试集。五篇合计 39 页；PDF、Docling 结果、模型和索引均为本地忽略资产，不提交。原解析为 Docling 2.48.0，`do_ocr=False`；已见的 v5 图表证据包含此前代理原页复核和特定修订，不代表任意 PDF 的自动解析可靠性。

## 数据、来源和索引

`scripts/docling_page_prepare.py` 逐篇校验既有 Docling 文本的 PDF SHA-256 与原 PDF，拒绝不一致或旋转页，渲染每页并保存页图哈希、论文与页码；页面只提供页面级位置。来源清单 SHA-256 `150ea56379e121fb985e6b5285808c7d27fc937ef983ed3b1d69618a074e914c`。随后把旧的 519 个 BGE 正文／图注向量从已校验 FAISS 复制到独立 Chroma，再只对 21 条可用表格文本编码，合并为一个共享 BGE+BM25 文本索引。表格文本移除可精确匹配的独立图注前缀，保留 `docling_table_text_unverified`、原表证据 ID、表级 bbox 和 PDF 哈希；不把表格解析当作已核对单元格引用。BGE 的最大输入为 768 token，长表密集表示可能截断，BM25 使用完整保存文本。

独立回读检查确认原始 519 条文本记录不变，迁移后向量逐元素最大差 `1.49e-8`，21 条表格文本与各自 Docling 来源和裁图哈希一致；这不是语义验证。`output/docling-table-index-verify-20261108.json` SHA-256 `1a24faaa3b7b2c739ff1f299990643e8ad34beec7ea6039107b901ba171b8624`。额外对 519 条原始文本、34 条原始图表记录核对原 PDF 哈希、页号、页面尺寸、bbox 边界和图表裁图哈希，553 条均通过结构检查；不证明每个区域的语义或多面板完整性。审计 SHA-256 `c488838577eb35dbca1d5fbb3ada88b2213c9bfae87afda7d34062c9030f551b`。

整页图像使用既有固定 CLIP 对齐快照，独立 Chroma 存储；不与 BGE 向量混算。CLIP 向量清单 SHA-256 `a8f50198d448a2b2db2ab36e15d713a5ce6df4fd9dc96fa810744c71a13a8682`，共享文本向量清单 SHA-256 `8b5631dcb54c936f759d9a757c432bf9581a7bf457eaf0619c504ed025f2a648`，文本／页面绑定文件 SHA-256 `f6649ae8f5136956a266d0a205e2cfb12ca934795f79b87717073bcee58017eb`。旧正文迁移约 2.17 秒，新增表格编码和索引写入约 17.11 秒，CLIP 页面编码与索引约 17.22 秒；它们来自不同进程、不是端到端时延。

## 真实查询和证据

`scripts/docling_clip_search.py` 对普通文本使用共享 BGE+BM25 排序与整页 CLIP 排序，按已验证 `(paper_id, pdf_sha256, page)` 去重，保留各原始文本证据 ID 与排名；缺少可信页面映射的文本单独列为 unknown。页面融合默认视觉权重为 0，恢复文本页面排序，不称作视觉增益；CLIP 排名仍单独返回。同页的表格／图片按各自独立的来源和哈希回溯并标 `same_page_only_unverified`，不由同页位置推断支持关系。

已见开发问题 “What are the FIGER and BBN accuracy scores for Ours (Full)?” 在 39 页上实际运行：目标论文 `1909.12079` 第 5 页的文本排名第 1、整页 CLIP 第 3、默认融合第 1；冷启动总耗时约 37.17 秒。返回的表格文本 ID `table-text-f7c7febdfbebe93ab608b29a` 标为未核实，表级位置 `[78.8827, 62.9648, 517.1000, 238.3535]`（top-left points）；独立表图 ID `v-29e405dbc6e7a1ebeb55ada0`，PDF SHA-256 `33826fdaafbc2b5e25e956dd0439261373a1e87f6a87313a41e81dcbb303fbcb`。助手同时查看原页和该表裁图，所问 Accuracy 分别为 75.5 和 82.5；这是单次代理复核，不是自动答案、独立人工金标或引用准确率。完整本地输出 SHA-256 `dbca2dfbe0242803b95259adfdbfb4a9d7569a97f93726852514f1e87bb2373d`。

另一条已见开发探针 “What are the two types of hints shown in Figure 3?” 未施加论文范围：`1909.06708` 第 4 页文本排名第 2、整页 CLIP 第 9、默认融合第 2，返回独立 Figure 3 图像 `v-f1f21457eabaec0c5e61d692`，标记同页关系未经语义核实；冷启动约 31.02 秒。助手查看原页与裁图，图例写有 hidden states 和 word alignments 两类 hints；该问题缺少论文范围，不能据单例证明一般情况下能消解同编号图的歧义。本地输出 SHA-256 `d26c82279042bec8a2c752352cf33c946c52758d8e824cb9256e91ab165ad84b`。此前误写为 “prompts” 的探索查询虽也返回该图，但图中标的是 hints，保留该结果，不把它作为语义正确的回答。

本轮没有调用远程生成模型，也没有绕过原 60 次验收预算；历史真实问答及其截断、误拒和复核限制见 [阶段记录](MULTIMODAL-PHASE1-STATUS-20261107.md)。该页面入口尚未接入既有 `ask`／`chat` 的生成与记忆流程，不能称为本轮新的端到端问答验收；原图输入、图片到文字和图片到图片功能只做已有回归，不以本样本改善其效果为目标。

## 复现

使用新输出路径；下列示例依赖已经下载的原 PDF、固定模型和旧 Docling 向量，既有来源与检查点不得覆盖。`docling-bge-NEW` 仅复制原向量，`docling-table-NEW` 复用这些向量并新增表格文本。

```powershell
$p = '.\.venv-gpu\Scripts\python.exe'
$v = '.\.venv-mm-20261030\Scripts\python.exe'
$env:TEMP = (Resolve-Path '.cache').Path
$env:TMP = $env:TEMP
& $p scripts/migrate_retrieval_to_chroma.py --kind bge --source indexes/multimodal-text-bge-20261030 --output indexes/docling-bge-NEW
& $v scripts/docling_page_prepare.py --text-index indexes/multimodal-text-bge-20261030 --pdf-dir data/raw/multimodal-eval --output data/raw/docling-pages-NEW
& $p scripts/docling_clip_pages_build.py --prepared data/raw/docling-pages-NEW --output indexes/docling-clip-NEW --visual-python $v --image-model models/multimodal-v1/clip-image-v1 --text-model models/multimodal-v1/clip-text-v1 --device cuda:0
& $p scripts/docling_add_table_text.py --text-index indexes/docling-bge-NEW --prepared data/raw/docling-pages-NEW --evidence data/raw/multimodal-eval/docling-v5-agent-reviewed/evidence.jsonl --evidence-root data/raw/multimodal-eval/docling-v5-agent-reviewed/crops --output indexes/docling-table-NEW --binding-output data/raw/docling-pages-NEW/binding-table.json --device cuda:0
& $p scripts/docling_verify_table_index.py --old-index indexes/docling-bge-NEW --new-index indexes/docling-table-NEW --evidence data/raw/multimodal-eval/docling-v5-agent-reviewed/evidence.jsonl --evidence-root data/raw/multimodal-eval/docling-v5-agent-reviewed/crops --output output/docling-table-verify-NEW.json
& $v scripts/docling_pdf_source_audit.py --prepared data/raw/docling-pages-NEW --text-index indexes/docling-bge-NEW --evidence data/raw/multimodal-eval/docling-v5-agent-reviewed/evidence.jsonl --evidence-root data/raw/multimodal-eval/docling-v5-agent-reviewed/crops --pdf-dir data/raw/multimodal-eval --output output/docling-source-audit-NEW.json
& $p scripts/docling_clip_search.py 'What are the FIGER and BBN accuracy scores for Ours (Full)?' --prepared data/raw/docling-pages-NEW --text-index indexes/docling-table-NEW --text-binding data/raw/docling-pages-NEW/binding-table.json --encoder 'models\bge-m3' --clip-index indexes/docling-clip-NEW --clip-python $v --clip-image-model models/multimodal-v1/clip-image-v1 --clip-text-model models/multimodal-v1/clip-text-v1 --evidence-jsonl data/raw/multimodal-eval/docling-v5-agent-reviewed/evidence.jsonl --evidence-root data/raw/multimodal-eval/docling-v5-agent-reviewed/crops > output/docling-clip-query-NEW.json
```
