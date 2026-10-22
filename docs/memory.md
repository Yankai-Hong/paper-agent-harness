# 三层记忆、归档召回与提交

近期 messages、结构化 SessionMemory、ContextArchive 三层结构保留。默认最多 8 条近期消息、40 facts、20 entities；同 key 事实更新覆盖旧值并保留原文证据 ID。语义模式额外生成 overview，事实选择与引文保留仍为确定性操作，不声称模型重写的事实一定正确。

`ContextArchive.search` 仅扫描相同 session 的校验过哈希的快照，支持 query、paper_ids、entity、fact_key、before_turn；按关键词相关性和 turn 顺序排序，默认最多 6 条、5000 字符。返回 archive_id、原 turn_id（旧归档可能缺失）、事实/消息、实体、证据 ID、superseded 标志。当前工作记忆优先于归档，同 key 的旧事实和含旧事实的助手消息标记过期；历史没有 turn_count 的旧归档顺序无法完整恢复。

Pipeline 在 NLU 前使用原问题召回必要上下文，在 NLU 后使用 standalone_query/论文范围再召回一次，不循环调用 NLU。召回记录进入 Trace，作为 historical_context_not_evidence 传给模型。Generate 的论文事实仍必须引用本轮有效 chunk，历史回答不能替代论文原文证据。--no-archive-recall 用于对照。

压缩前完整状态归档；语义摘要失败则 deterministic fallback，记录 degraded 和原因。归档写入失败时本轮失败并回滚内存状态，不能声称可靠提交。bounded 压缩记录前后字符、字符估算 token、保留实体/事实和原样保留一致性；它不是语义正确率，实际 token 由 API usage 单独汇总。

`PaperPipeline.run(question, state, turn_id=...)` 接受稳定 ID。已提交相同 ID/问题直接返回已存答案，不再调用 API 或追加消息；同 ID 不同问题拒绝。每轮在副本回滚边界内执行，成功把消息、事实、计数、committed_turns 一次原子替换写入 `<session>.state.json`；load 优先新快照，兼容旧独立文件。SessionMemory.load 也读取快照。提交前失败不残留半条消息，重试 Generate 不写记忆。

局限：同 session 多进程并发写不受锁保护；调用方应串行执行一个会话。失败调用可能仍产生服务端成本。committed_turns 的结果账本随会话增长，不进入模型上下文但需要长期保留策略。归档线性扫描适合当前本地规模，尚无语义索引。只保存旧快照，不自动把旧 facts 全部写回工作记忆。

真实长会话消融和故障注入分开报告，见 `reports/MEMORY_EVAL_RESULTS.md` 与 `reports/RELIABILITY_RESULTS.md`。新版非空检查点评估讨论论文与事实来源，不能冒充事实文本正确性、冲突解决或人工语义一致性金标。
