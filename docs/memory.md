# 工作、情景与长期记忆

架构只有三类记忆。论文 chunks 和本轮可引用证据属于独立 FAISS 知识库；ContextArchive 是会话快照和审计恢复机制，不是第四层记忆。

| 类别 | 权威内容 | 可重建投影 |
|---|---|---|
| 工作记忆 | SQLite `sessions` 中近期消息、实体、未解决任务、摘要、turn 账本和已提交归档 ID | Redis TTL 缓存及 state.json 文件 |
| 情景记忆 | SQLite `episodes` 保存重要讨论、决策过程、实验结果、待办原文 | Chroma 语义索引；索引损坏或不可达时有标记的词项降级 |
| 长期记忆 | SQLite `long_term` 保存明确用户偏好、约束和稳定决策，含 scope、subject、attribute、kind、status、source、version、provenance_refs、replaces | 无额外事实层 |

默认 `sqlite_local` 使用相同三类逻辑和 SQLite 权威提交，未启用 Redis/Chroma 投影；不是“真实三服务集成”。`MEMORY_BACKEND=three_layer` 使用已有 BGE 编码器、Redis 和 Chroma。旧 `legacy` 配置值也走 SQLite 本地模式，不能继续使用旧事实运行链。FactStore、put_fact、update_fact_memory 与 SessionMemory.facts 已退役。旧 facts.sqlite3 仅由迁移工具只读打开，不参与正常召回。

## 身份和共享

当前是本机 CLI，没有登录认证。`MEMORY_LOCAL_USER` 默认 local，`MEMORY_PROJECT` 默认由 memory 根目录绝对路径确定；它们是操作者声明的本地命名空间，不能当成安全认证。指定用户/项目时 SQLite 与归档放到 `scopes/<identity hash>`；Redis key 和 Chroma collection 同样加命名空间。不同操作系统使用者仍应依靠文件权限保护本机数据。

默认长期记忆作用域为 `session:<id>`，项目内不同会话也不共享。只有显式 `MEMORY_SHARE_PROJECT=true` 才将新长期声明及重要事件标记为项目共享，并允许新会话恢复它们；此前私有会话记录不自动扩大范围。共享限于相同本地用户、项目和存储根目录。关闭后停止跨会话读取，不删除已有共享记录。未实现跨项目用户级全局共享，以免默认串读。

明确声明语法：`偏好:语言=中文`、`约束:资源=单卡`、`决定:索引=FAISS`、`记住:键=值`；支持 Remember/Preference/Constraint/Decision。`更新记忆:键=值` 显式替代同范围、主体、属性的活动版本；旧版本 superseded。重复值不新增；没有更新标记的同属性不同值进入 conflict。不同属性不因文字相似误判冲突。模型推断不自动 active。普通论文 Claim 不保存为长期记忆，即使逐字引用原文。

`实验结果:`、`研究决策:`、`TODO:`、`后续待办:` 等进入情景记忆。标记规则刻意保守，会漏掉自然语言表达；不是通用语义抽取。长期记录默认召回 active/conflict；superseded 仅在明确询问历史时进入上下文，candidate/archived 不召回。`MemoryStore.delete(session_id,memory_id,shared=False)` 是保留审计的逻辑删除；项目记录须 shared=True。不是隐私硬删除，历史事件、归档及备份仍需单独管理。

## 调用与恢复

Pipeline 在 NLU 前后两次召回，后一次以 standalone_query 排序；显式偏好/约束不受论文范围过滤误伤。不同来源使用相关性档位及来源内 RRF 名次，不能把词项覆盖率当成余弦概率。默认最多6条/5000字符。Chroma 命中必须与 SQLite 原始事件指纹一致；缺失或损坏标记 indexed=0，最多32条批量核查与修复。跨会话共享事件当前用 SQLite 词项召回，不伪称跨会话 BGE 向量召回。

所有历史上下文标记 `historical_context_not_evidence`；Generate 仍只引用本轮有效 chunks。归档只回流历史消息，旧事实条目不进入召回。旧摘要保留并标记 legacy_context_unverified，不能当作已确认长期记忆；摘要可能已包含旧事实，不宣称已消除所有历史文字。

Memory 阶段在一个 SQLite BEGIN IMMEDIATE 事务内写长期版本、事件原文、会话快照和 turn 账本。稳定 turn_id 已提交时直接重放答案，相同 ID 不同问题拒绝。归档原文先写；只有归档 ID 出现在已提交快照时可召回。崩溃留下的未提交归档保留但不可召回。Redis/Chroma/文件同步在提交后补偿，失败不会回滚已提交答案。同 session 版本检查阻止过期并发写。

压缩保留近期8条消息、20实体和全部未解决任务；semantic 将上次 overview 与上次摘要轮次以后的消息合并，论文顺序来自实际引用。旧无 turn_id 数据只能按可见消息重新摘要。失败回退 bounded；摘要调用计入本轮预算。SQLite 事务持锁时摘要仍可能等待模型，是本地部署限制。

## 迁移与回滚

先停止向目标目录写入，盘点来源归属；不要在旧代码与新代码间同时写同一目录。

```powershell
$py = '.venv-gpu/Scripts/python.exe'
& $py scripts/migrate_memory.py runs/memory --output runs/recovered/memory --confirm-local-owner
# 不确定归属则省略 --confirm-local-owner，长期记录不会晋升。
& $py scripts/rollback_memory.py runs/memory/migration-backups/v1-facts.sqlite3 runs/rollback-memory
```

首次迁移用 SQLite backup 生成只读副本；原库不修改、不删除。旧独立会话/记忆文件也先逐文件备份。未确认归属时，会话、事件和旧事实均只留备份，不自动导入；确认后须选择新的空目标，不能把旧未确认迁移记录直接升级。版本1/2 审计与逐记录结果在目标 migrations 表和审计文件，重复执行返回同一结果。仅能由原 turn 账本核对属性和值的 active/conflict 用户声明迁入长期表；paper_evidence、不明来源、旧 superseded 版本仍留备份，不自动晋升。确认归属后恢复事件原文和会话快照；快照中的 facts 字段剥离，旧账本可重放，历史答案不是本轮证据。旧无账本文件可恢复近期上下文，但不会因此自动晋升事实或授权归档。

要使用上例的迁移目标，在同一工作区从 `--state-dir runs/recovered` 启动；它使用 `runs/recovered/memory/memory.sqlite3`。根据实际需要先复制原归档目录到该目标的 memory/archives，归档仍须通过提交账本授权。

回滚工具只创建新的空目录，恢复旧 facts.sqlite3 与旧会话快照，不覆盖新数据。保留整个新版目录以保存迁移后的新增轮次；用基线 Git 版本运行隔离的 rollback-memory。若要恢复旧归档，另复制原目录 archives 到回滚目录，不能以文件存在替代提交授权。切回旧代码会恢复旧架构行为，只用于审计/回滚；正常运行应使用新版本。

本次真实历史资产迁移演练和计数见 [stage_c](../reports/revision/stage_c.md)。数据库/原文不提交 Git。历史 final-acceptance 是旧版本成绩，不套用到当前实现。
