# 答案生成与引用

实现：[providers.py](../src/paper_agent/providers.py) 的 generate schema 和 [validate_citations](../src/paper_agent/memory.py)。

生成器接收当前问题、检索证据与明确标记的历史上下文，不以历史回答当作新事实证据。返回 claims，每条有稳定 key、text 和一个或多个 `paper_id/chunk_id` 引用。不提供额外的无引用自由文本字段，从结构上减少未附来源的结论。

```json
{"claims":[{"key":"retrieval-method","text":"论文采用……","citations":[{"paper_id":"p1","chunk_id":"c1"}]}]}
```

校验要求每条 claim 有正文、有引用，引用 chunk 必须在本次证据中，且其 paper_id 匹配。成功后返回 decision=answer。**校验失败时不再整轮报错**，而是转为受控拒答：`decision=refuse` 并记录 `reason_code=citation_validation_failed`；无效引用不会被静默替换成其他引用。空 claims 属 schema 违例，仍终止本轮。

拒答和澄清使用本地模板，不额外调用生成 API。CLI 直接展示结构化答案及 trace_id，便于检查原文与调用记录。

**存在引用不等于引用支持结论。** 当前校验不能识别断章取义、数值偷换或过度概括。引用语义支持与答案正确性必须独立标注；评分入口要求审阅者给出明确判断。没有以引用结构通过率冒充截图中的引用准确率。

## 本轮结构修复与语义评估

Provider 在非法 JSON/Schema/截断时最多一次要求更短合法输出的修复。Generate 可接收标记为非证据的历史上下文，论文主张仍只引用本轮 evidence。引用失败拒答记录 degraded；成功 trace 分开记录 id_valid/source_consistent，semantic_support 不由结构校验推断。独立 judge_acceptance 保存逐 claim 支持理由及原始证据，结果为 provisional，需人工抽查。
