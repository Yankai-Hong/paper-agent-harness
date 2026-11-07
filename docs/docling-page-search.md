# Docling PDF 页面检索

`scripts/docling_clip_search.py` 对已有 Docling 文本和原 PDF 渲染页面分别使用共享 BGE+BM25 与独立 CLIP 索引。文本、图表裁图和整页分别保留各自来源；页级合并不证明内容之间存在支持关系。默认 CLIP 融合权重为 0，CLIP 排名仍单独返回。具体准备、真实小样本、复现命令和限制见 [运行记录](../reports/revision/DOCLING-CLIP-PAGE-20261108.md)。

表格文本标记为 Docling 提取且未经核实，表级 bbox 不是单元格坐标。没有可信页映射的文本保留为 unknown。该页面检索入口尚未替换 `ask`／`chat` 的证据选择，生成模型问答的既有预算和来源限制继续适用。
