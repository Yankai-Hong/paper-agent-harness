# 语料与分块

实现：[corpus.py](../src/paper_agent/corpus.py)。来源：PDF 文本提取使用 [pypdf](https://github.com/py-pdf/pypdf)，正式 tokenizer 使用 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)。分块和字符溯源是本项目领域代码。

输入 JSONL 每行是一篇论文：

```json
{"paper_id":"p1","title":"Example","source_url":"https://example.org/p1","paragraphs":[{"paragraph_id":"s1-p1","section":"Methods","text":"Original paragraph.","page":1}]}
```

先按章节和段落组织，短段落在同一章节内打包；超长段落使用 token 滑窗。正式默认 500 token，重叠 80 token。每个 chunk 保存稳定 chunk_id、paper_id、章节、原文文本、段落 spans 和字符起止位置。测试逐一核对字符切片等于原始段落内容。没有原始页码时不补造页码。

```powershell
& $py -m paper_agent.cli pdf paper.pdf --output data/paper.jsonl
& $py -m paper_agent.cli ingest data/paper.jsonl --output data/chunks.jsonl --tokenizer models/bge-m3
```

不传 tokenizer 时使用 lexical-v1，只适合轻量基线/安装检查。不能把词法分块与 BGE 分块结果混做消融。PDF 双栏顺序、章节识别、表格需人工检查；扫描件不支持 OCR。

实测语料为官方 QASPER v0.3：100 篇全文，形成 1,996 个 BGE chunks。`prepare_qasper.py` 使用 seed=42 固定抽样并保存 archive SHA256；只索引正文/摘要，不索引答案。官方证据必须唯一匹配原文段落，不能映射的样例被排除并记账。详见 [评测说明](evaluation.md)。

## 本轮资源与标签版本

直接复用原 data/qasper、models 与 indexes/bge，未重建语料或下载模型。修订评测集为 data/eval/v0.3，manifest 保存来源与哈希；原 data/eval 文件保留。人工复核状态不由静态检查或 LLM Judge 提升。

本轮实际 API 数据冻结为 v0.3；后续 v0.4 补充显式 history session/turn 元数据、论文 ID/标题别名和实体指标适用范围。v0.4 的 Rewrite 分数是对保存的 v0.3 预测离线重算（`reports/acceptance/rewrite/rescored-v04.json`），不是新 API 实验；两版均保留草稿标签与原始结果。
