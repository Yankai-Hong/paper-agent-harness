# Phase A 数据审计

审核为原始材料与规则辅助核对，没有外部人工标注者。human-gold.jsonl 为空；source-verified 与 needs_human_review 独立保存。

| 数据集 | 前/后样本数 | 来源规则/官方标注可确认 | 待人工 |
|---|---:|---:|---:|
| intent | 100/100 | 63 | 37 |
| query_rewrite | 90/90 | 80 | 10 |
| boundary | 55/55 | 20 | 35 |
| dialogs | 120/120 | 0 | 120 |
| memory | 80/80 | 0 | 80 |
| retrieval | 100/100 | 100 | 0 |

Rewrite expected_query 缺失由 90 降至 0：60 条补齐论文/问题，20 条已独立问题原样保留，10 条显式歧义问题保留澄清需求。参考不是唯一措辞；token overlap 仅为词面诊断。
Intent 类别分布未修改。v0.5 开发审核曾错误地按轮次推断 intent，已标记 SUPERSEDED，未用于任何评测；v0.6 只依据明确任务动词，保留不确定类别。
每条原值、建议值、原因、来源问题/论文/证据 ID、状态见 data/eval/v0.6/audit.jsonl；545 条审核记录包含不同任务对同一问题的视图，不能当作 545 个独立会话。
Memory 80 条非空检查点覆盖讨论论文和来源保留；自然事实更新、冲突及归档精确率缺少可靠期望，显式 not_applicable，未编造金标。
Retrieval 使用官方 QASPER 有文本证据的 100 条；原始筛选排除 60 条无答案/无文本证据、126 条证据无法映射，见 data/qasper/manifest.json，不纳入可回答 Recall 分母。
Claim 源文诊断 8 条见 phase-a/claim-benchmark/cases.jsonl 和 data/eval/claim-diagnostics-v1；预期由代理核对原文，不是独立人工金标。正式人工 Claim 验证仍缺失。
进一步逐条读取原始 QASPER 压缩包中的全部标注：100 条中有 12 条可回答性分歧、2 条 yes/no 分歧，没有一致标为不可回答的样本。完整原始答案、证据与复核队列见 phase-a/source-annotation-audit。原检索协议选择至少一个可映射的可回答标注，并不代表所有标注者一致；100 条主分母不因观察到模型结果而删改。
