# 三层记忆、归档召回与提交

## MemoryManager 存储模式（2026-10）

`MEMORY_BACKEND=three_layer` 启用 `MemoryManager`；默认 `legacy` 保留已有离线评测与旧状态行为。Pipeline 可以直接注入 `memory_manager`。本节描述新模式，后文原文件模式仍适用于 legacy。

| 层 | 存储 | 内容与权威边界 |
|---|---|---|
| Working | Redis JSON，`paper-agent:working:v1:<session_id>` | 最近 8 条消息、活动实体/任务摘要、最多 40 个有效事实、未解决问题；默认 TTL 86400 秒 |
| Fact | `<memory_dir>/facts.sqlite3` | 事实版本、来源、状态；同时保存完整提交快照、turn 账本和情景原始记录 |
| Episodic | `<memory_dir>/chroma`，独立 collection | BGE-M3 情景向量索引；原始事件保存在 SQLite，可重建索引 |

调用链：`PaperPipeline.run → load_working_memory` 恢复持久状态并补偿未同步索引；NLU 前按原问题召回，NLU 后按 standalone_query/论文范围召回；Memory 阶段调用 `commit → extract_memories → update_fact_memory / write_episode → compact_working_memory`，在一个 SQLite 事务内提交事实、情景原文和状态。提交后更新文件快照、Redis、Chroma。七阶段名称和 FAISS 论文检索不变。

Redis 不是唯一副本。缓存过期或不可用时从 SQLite 恢复，文件快照保持旧入口可读。Chroma 失败保留 `indexed=0` 待同步事件，每次恢复最多重试 32 条，upsert 使用稳定事件 ID。读取降级为 SQLite 原文的词项匹配。`MEMORY_ALLOW_FALLBACK=false` 在提交前读/相似搜索失败时明确失败；提交后缓存或索引故障不能回滚已经成功的 SQLite 事务，返回 committed + degraded，下一请求恢复。SQLite 或压缩归档在提交前失败，则回滚事实、事件、状态。本实现不提供跨存储分布式事务。

同 session 并发提交通过 SQLite `BEGIN IMMEDIATE` 和版本比较拒绝过期写入；失败方应使用同 turn_id 重新加载并重试，可能已有重复 API 成本，但不会重复提交记忆。已提交 turn 的重放不再生成答案。Redis 不保存无限增长的结果账本；SQLite 和文件保存账本，尚未实现账本保留期限。文件投影短暂落后时，三层模式必须以 SQLite 为准，不能直接切换 legacy 写入同目录。

## 事实提取、版本及证据限制

Canonical key 使用 NFKC、大小写和分隔符规范化，并加入 source_type、paper_id 作用域。相同 key/内容不新增；不同内容默认 conflict。只有显式替代当前 fact_id 的更新才将旧版本 superseded。`记住:键=值`、`偏好:键=值`、`约束:键=值`（或 Remember/Preference）保存明确用户声明；`更新记忆:键=值` / `Update memory:key=value` 明确替代已有用户事实。普通新说法和模型自报置信度不会触发覆盖。不同 key 的高词项重合内容标记潜在冲突，不合并。删除是带时间戳的 deleted 状态，保留历史。

论文候选写入前重复调用 `validate_citations`，所有 chunk_id/paper_id 必须属于本轮真实 evidence。ID 正确不等于语义正确：目前仅与每个引用 chunk 的完整句子或全文一致的原文摘录进入 active；改写、数值不一致、无法确定的推断保存为 `conflict / semantic_support_unverified`。这是保守确定性规则，不是经过人工验证的通用语义判定器，也不证明论文自身结论的真实性。该限制会产生漏收，不使用同模型 Judge 自动提升权威事实。

情景只保存用户显式标记的持续价值交互，如 `研究决策:`、`实验结果:`、`方法变更:`、`后续待办:`、`未解决问题:`、`复用笔记:`、`研究约束:`、`偏好:键=值`（同时支持对应英文标记）。普通问答即使有 Claims，也只保留在对话/归档中，不默认创建长期情景。相同事件 ID 幂等，完全相同内容去重；相似事件记录关联 ID，不能单凭向量相似性删除或覆盖。不同会话、论文、event_type 在 Chroma 查询时过滤；返回最多 6 条且序列化总长不超过 5000 字符。这是字符预算，不冒充实际 token 数。活动事实和历史事件分别带 kind；所有召回仍通过 `historical_context_not_evidence` 输入 NLU/Reflect/Generate，论文答案仍需本轮 FAISS chunk 引用。

## 召回与索引修复（2026-10-26）

`retrieve_memories()` 将 SQLite、Chroma、归档候选统一为带 `record_id`、`kind`、`session_id`、`paper_ids`、`status`、`source_type`、时间、证据 ID 和分数类型的记录。先分相关性等级，再比较来源内名次的 `1/(60+rank)`；状态、来源可信度和时间只用于后续排序。词项覆盖率与余弦分数不当成同一种概率直接混排。词项覆盖率低于 0.2 不进入候选，高相关档为 0.6；语义高相关档为 0.65。这些是透明的工程阈值，尚未做人工语义标注校准。

同事件、已存在于 SQLite 的事实与其归档副本去重；SQLite 状态优先，不能通过归档将 deleted 或 conflict 重新当成 active。混合来源时，SQLite 初始占额不超过返回数量的一半，并为能放入预算的非 SQLite 候选预留字符空间；剩余容量再补事实。全部记录仍受默认 6 条/5000 字符限制；大于总预算的单条记录不会强行塞入上下文。NLU 后使用 standalone_query 重新排序的结果，不再让原问题的旧候选先占满预算。

Chroma 显式使用单位归一化向量。默认 `hnsw:space=l2` 是平方欧氏距离，余弦为 `1-distance/2`；已有 cosine collection 使用 `1-distance`。读取实际 collection 度量，不猜测未知类型。`MEMORY_MIN_SIMILARITY=0.35` 是默认最低余弦要求（范围 0..1）。空查询、空库返回空列表；缺失/非法距离、零向量、NaN/Inf 或未知度量明确报错，由 manager 按配置做有来源标记的词项降级，不伪造语义分数。旧 BGE 索引原本已使用单位向量，缺少 hnsw:space 元数据按 Chroma 默认 l2 解释。

事实的 `paper_ids` 统一解析为列表，兼容旧单个 paper_id 与 JSON 数组字符串；按集合交集过滤，多论文事实可被任一关联论文召回。非法旧格式明确报错。SQLite 保留原 paper_id 列以兼容已有数据，不重建事实数据库。

active 事实可作为当前事实候选；conflict 以 `kind=conflict`、`requires_resolution=true`、`is_current_fact=false` 返回；superseded 默认排除，仅在明确询问历史/之前/变更等问题时召回。所有记录均携带 `historical_context_not_evidence=true`、`is_authoritative=false`，这里的非权威指不能充当 Generate 的本轮论文证据。召回不会修改事实状态，更不会自动晋升冲突。

索引检查使用 SQLite 中持久化的 session 游标，每次最多核对 32 条事件 ID/内容指纹；没有缺失也推进游标，到末尾后循环。缺失、旧版本、缺少指纹的旧索引均设 indexed=0，复用现有最多 32 条的同步流程。Chroma 故障保留待同步状态且不推进游标。写入前清除同 ID 旧向量元数据，再幂等 upsert，防止旧论文过滤字段残留；中途失败仍有 SQLite 原文可恢复。查询返回的指纹与 SQLite 不符时拒绝使用并排队修复，孤立索引记录不会覆盖原文。低相关查询没有命中并不代表索引缺失。

检查分别由恢复和上下文读取触发；大于一个批次的缺失可能需要后续调用才能全部发现。分批限制适用于一致性核查；当前事实/情景原文的候选装载、归档读取与完全重复内容检查仍适合本地规模，未引入大规模数据库全文检索。语义阈值与显式事件标记会漏掉某些有价值的自然表达，这是保守筛选的已知限制。

## 配置、启动与兼容

```powershell
.venv-gpu/Scripts/python.exe -m pip install --no-cache-dir -c constraints-memory.txt -e '.[memory]'
docker run -d --name paper-agent-redis -p 127.0.0.1:6379:6379 redis:7-alpine
$env:MEMORY_BACKEND='three_layer'
$env:REDIS_URL='redis://localhost:6379/0'
.venv-gpu/Scripts/python.exe -m paper_agent.cli chat --index indexes/bge --mode hybrid --encoder models/bge-m3 --device cuda:0 --session memory-demo
```

Docker 服务需要已启动；也可直接使用现有 Redis。SQLite/Chroma 是本地持久存储，不另起数据库服务。没有自动下载 embedding：显式复用 `retriever.encoder.encode()`，须使用现有本地 BGE 编码器；当前 CLI 的 bm25 模式没有 encoder，启用三层模式会明确报错，应使用 dense/hybrid/rerank。无需改动真实 API 凭据。

环境选项见 `.env.example`：`MEMORY_WORKING_TTL=0` 禁用缓存过期；`MEMORY_REDIS_NAMESPACE` 隔离应用；`MEMORY_CHROMA_PATH` 和 `MEMORY_CHROMA_COLLECTION` 控制索引。SQLite 路径随现有 memory_dir。collection 保存模型路径、max_length、可用 provenance 的签名，模型改变拒绝混用；新 collection 可从 SQLite episodes 重新写入。不能将论文 chunk 放入情景 collection。

旧 state.json/history.json/SessionMemory 继续可读，首次成功提交将已有事实作为 conversation/conflict（legacy_unverified）导入，不凭历史 ID 自动认定论文证据有效。旧 ContextArchive 不搬迁、不删除，仍可按需词项召回。压缩仍使用原 compact_state / semantic_compact_state，先保存完整归档；语义摘要失败回退确定性裁剪。长期 SQLite 事实/事件不随工作上下文裁剪删除。当前没有自动人工审核队列、跨 session 共享事实或通用语义冲突裁决。

存储接口依据 [Chroma Python reference](https://docs.trychroma.com/reference/python) 与 [Redis 客户端超时文档](https://redis.io/docs/latest/develop/clients/redis-py/produsage/)，实际版本和测试记录见 `reports/history/development/MEMORY_UPGRADE_REPORT.md`。

## 原文件模式（legacy）

近期 messages、结构化 SessionMemory、ContextArchive 三层结构保留。默认最多 8 条近期消息、40 facts、20 entities；同 key 事实更新覆盖旧值并保留原文证据 ID。语义模式额外生成 overview，事实选择与引文保留仍为确定性操作，不声称模型重写的事实一定正确。

`ContextArchive.search` 仅扫描相同 session 的校验过哈希的快照，支持 query、paper_ids、entity、fact_key、before_turn；按关键词相关性和 turn 顺序排序，默认最多 6 条、5000 字符。返回 archive_id、原 turn_id（旧归档可能缺失）、事实/消息、实体、证据 ID、superseded 标志。当前工作记忆优先于归档，同 key 的旧事实和含旧事实的助手消息标记过期；历史没有 turn_count 的旧归档顺序无法完整恢复。

Pipeline 在 NLU 前使用原问题召回必要上下文，在 NLU 后使用 standalone_query/论文范围再召回一次，不循环调用 NLU。召回记录进入 Trace，作为 historical_context_not_evidence 传给模型。Generate 的论文事实仍必须引用本轮有效 chunk，历史回答不能替代论文原文证据。--no-archive-recall 用于对照。

压缩前完整状态归档；语义摘要失败则 deterministic fallback，记录 degraded 和原因。归档写入失败时本轮失败并回滚内存状态，不能声称可靠提交。bounded 压缩记录前后字符、字符估算 token、保留实体/事实和原样保留一致性；它不是语义正确率，实际 token 由 API usage 单独汇总。

`PaperPipeline.run(question, state, turn_id=...)` 接受稳定 ID。已提交相同 ID/问题直接返回已存答案，不再调用 API 或追加消息；同 ID 不同问题拒绝。每轮在副本回滚边界内执行，成功把消息、事实、计数、committed_turns 一次原子替换写入 `<session>.state.json`；load 优先新快照，兼容旧独立文件。SessionMemory.load 也读取快照。提交前失败不残留半条消息，重试 Generate 不写记忆。

局限：同 session 多进程并发写不受锁保护；调用方应串行执行一个会话。失败调用可能仍产生服务端成本。committed_turns 的结果账本随会话增长，不进入模型上下文但需要长期保留策略。归档线性扫描适合当前本地规模，尚无语义索引。只保存旧快照，不自动把旧 facts 全部写回工作记忆。

真实长会话消融和故障注入分开报告，见 `reports/history/development/MEMORY_EVAL_RESULTS.md` 与 `reports/history/development/RELIABILITY_RESULTS.md`。新版非空检查点评估讨论论文与事实来源，不能冒充事实文本正确性、冲突解决或人工语义一致性金标。
# 归档提交边界（2026-10-26 修复）

压缩仍先保存不可变原文，再裁剪工作上下文。新归档含 `commit_protocol=1`，其 SHA-256 ID 加入候选状态的 `archive_ids`。只有该 ID 出现在持久化提交快照中，`ContextArchive.search` 才允许召回；文件存在本身不构成已提交证明。legacy 的提交点是原子替换 `.state.json`；three_layer 的提交点是 SQLite 同事务写入事实、事件及 session 快照，召回授权直接从 SQLite 获取，不信任 Redis 或文件投影。

提交失败、进程在写归档后退出时，原文文件保留但不可召回；不依赖异常清理。相同 turn_id 重试仍走已提交轮次账本。旧归档不删除：无协议字段的归档须逐条与持久化轮次账本的原问题/答案及事实文本/引用核对；证据不足的历史保留在磁盘，可按 ID 检查，但不会自动进入模型上下文。这是保守兼容，并非保证所有无账本旧数据可自动恢复。

论文事实身份由 source_type、排序后的论文范围、canonical attribute/key 组成。可提供 `attribute` 作为稳定可比较属性；同论文同属性变化须显式 `replaces`，保留 superseded 旧版，否则记录 conflict。不同论文或不同论文属性不再仅因文本相似而冲突。user/conversation 的旧保守相似规则仍存在；不同来源互相隔离。本规则不理解任意自然语言属性同义词。

`archive_ids` 和幂等账本随会话增长；归档扫描仍是本地规模线性读取。语义摘要发生在 SQLite 写事务内，可能持锁等待模型；本轮不扩展为后台任务或生产并发系统。API timeout 不是宿主暂停、DNS 和所有服务端状态下的绝对墙钟保证。
