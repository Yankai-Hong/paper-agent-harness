# v0.4 修订数据

原始问题、会话与 turn 不变。相对 v0.3：history 明确携带 session_id/turn；Rewrite 增加 required_entity_aliases 和 entity_preservation_applicable。t2/t3 共60个实体适用样本，全部90条仍保留。expected_query 未人工标注仍为 null，不填入模型自身输出作为 gold。

Memory 是8组十轮开发会话，非空检查点检查讨论论文和事实证据来源；不是人工语义事实金标。全部派生标签仍为 draft_unreviewed。官方 QASPER retrieval 单独保持来源标注。

v0.3 已完成实际 API 实验；v0.4 的 alias 分数仅离线重算旧预测，不能声称模型又跑过这一版。数据生成器拒绝覆盖有 manifest 的输出目录。
