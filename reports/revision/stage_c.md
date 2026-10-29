# C：工作、情景与长期记忆

移除 FactStore 和 SessionMemory.facts，普通论文 Claim 不再自动长期保存；SQLite `memory.sqlite3` 承担 session 快照/turn 账本、重要事件和有作用域的明确长期声明，Redis TTL 与 Chroma 是可补偿投影。ContextArchive 仍为经提交授权的审计快照，不是第四类记忆。paper_id+chunk_id 联合校验本轮证据。语义压缩合并旧摘要和新增消息，旧摘要可能包含未经确认的历史信息，标记来源而非宣称不存在。

本机命名空间由 local user、project 和根路径组成，不具备认证。默认 session 隔离，只有显式项目共享的新声明/事件才能跨会话恢复；不同用户、项目、根目录和同名 session 隔离。逻辑删除、冲突、显式更新和版本保留见 docs/memory.md；归档、备份及历史消息不是隐私硬删除。

## 实际盘点和迁移演练

baseline.json 盘点37个 SQLite 文件，其中21个是旧 facts.sqlite3，均在历史评测目录；没有在 runs 下发现旧 SQLite 事实库。原始21库合计922条：904条 paper_evidence、18条用户显式评测声明。对明确属于历史评测的库在独立目标执行版本1迁移并重复执行：94个会话、54事件、12条声明迁入；904论文证据和6条旧 superseded/未核实版本留只读备份，不参与活跃召回。原库哈希逐个不变。逐库、路径和计数见 migration-inventory-v2.json；每库备份位于源目录的 migration-backups/，逐记录审计位于 migration-revised/ 的 migration-v1-audit.json。

另在 runs 发现46对独立 history/记忆 JSON（92文件、483个旧事实条目，可能有重复），来源归属没有逐个确认。版本2只读备份92文件，46个会话全部保留在备份，没有导入或晋升；file-memory-inventory.json、file-migration-inventory.json 只含路径、哈希和计数。不能声称没有个人旧数据。确认归属后可迁到新的空目标；无法与提交账本核对的声明保持未晋升。

命令与隔离回滚详见 docs/memory.md 和 scripts/migrate_memory.py、scripts/rollback_memory.py。旧库/文件和历史报告未覆盖或删除；migration-inventory.json 是第一版演练，修订后的权威清单为 migration-inventory-v2.json。备份和数据库属于本地数据，不随仓库提交。

## 验证与实验

最终全量151 passed，73条 Chroma/Pydantic 弃用警告。真实 FAISS/BGE/Cross-Encoder + Redis + SQLite + Chroma 六轮，固定回答器、零模型请求：TTL过期恢复、幂等重放、Chroma部分删除修复通过；integration-final-v2/result.json。首次 Redis 未启动导致的失败保留 integration-v1；测试端口16487为临时实例，已关闭。迁移、归属、重复执行、隔离、冲突、删除、崩溃和提交失败等为自动化回归，不混同真实模型质量。

固定规则标签10例（4个正例），新写入 precision/recall 均4/4；旧门禁的确定性重放 precision 为3/13，新为4/4。4个授权跨会话属性的 Recall@6 为4/4。样本与规则同源且规模极小，只验证机制，不是自然语言泛化或人工金标。真实跨轮/跨会话指代、压缩语义保真为 unknown；见 experiments-final/memory.json。
