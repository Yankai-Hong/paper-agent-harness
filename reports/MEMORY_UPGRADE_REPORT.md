# 三层记忆升级验收记录

## 2026-10-26 剩余问题修复验收（当前实现）

本节更新当前实现结论，下方 2026-10-25 结果作为历史记录保留，不代表本次重跑。Git 基线仍为 `4368a62a4ba15160fb5ff452194fd10bd2737c4e`，工作区已有未提交修改均保留。

### 文件与实际修复

- `memory_manager.py`：`retrieve_memories()` / `rank_memories()` 将事实、情景、归档候选统一状态、来源、论文范围及评分字段，按相关性等级和来源内名次排序，再去重和施加数量/字符预算。混合来源限制 SQLite 初始占额，并预留可容纳的非 SQLite 候选字符空间；空余容量再补事实。`extract_memories()` 不再因为存在 Claims 就保存情景，只识别明确决策、偏好、实验结果、变更、待办、复用笔记等用户标记。`check_episode_index()` / `repair_projections()` 分批核对和补偿索引。
- `memory_stores.py`：`paper_ids()` 兼容旧标量、JSON 数组字符串及新列表，读写统一后按集合交集过滤。`FactStore.audit_batch()` 使用持久 session 游标和 SQL LIMIT；`ChromaEpisodes.vectors/search/fingerprints/write()` 验证并归一化向量，返回实际距离/度量/相关性，保存原文指纹，修复部分丢失或过期元数据。
- `pipeline.py`：NLU 后使用独立查询重新统一排序的候选，不再让 NLU 前列表优先占满上下文。仍通过 historical_context_not_evidence 输入 NLU、Reflect、Generate。
- `.env.example`：新增 `MEMORY_MIN_SIMILARITY=0.35`（最低余弦阈值，范围 0..1）。
- `tests/test_memory_recall_fixes.py` 新增 14 个必要测试实例；`tests/test_memory_manager.py` 保留原断言，适配带分数的 Chroma 返回和显式研究决策输入。`scripts/check_memory_integration.py` 同步适配显式决策/event_type；本轮未重跑该 Redis/BGE 脚本，旧真实集成数据不冒充新结果。
- `docs/memory.md` 更新当前排序、过滤、状态、写入筛选与恢复规则；本报告保留历史并添加本次结果。

### 评分、安全与恢复边界

Chroma 本地源码默认 hnsw:space 为 l2；其距离是平方欧氏距离，单位向量换算余弦为 `1-d/2`。已有 cosine collection 使用 `1-d`。真实临时 Chroma 测试同时验证两种度量；低相关、空查询/空库、非法或缺失距离、NaN/Inf/零向量均有明确处理。异常可按现有配置降级为真实词项匹配，不虚构语义分数。低相关空结果不触发“索引丢失”判断。

词项覆盖率与余弦不是同一种概率：只比较相关性档位、来源内名次 `1/(60+rank)`，再以状态、来源、时间排序。默认词项最低 0.2、高相关 0.6；语义最低 0.35、高相关 0.65。阈值是工程策略，未宣称经过人工语义金标校准。

active 是当前事实候选；conflict 保留 kind=conflict、requires_resolution=true、is_current_fact=false；superseded 仅在明确历史变更查询下召回。所有候选保留非论文证据标记，冲突不会晋升 active。测试直接检查三个模型阶段的输入，确认冲突没有变成 Generate 的当前 evidence。原 validate_citations 与逐字摘录支持规则未放宽；多论文事实可被任一关联论文召回，无关论文不会命中。

索引核查每次最多 32 条，以 SQLite 稳定 ID/原文指纹为准。查不到 ID、旧指纹或旧格式无指纹都排入 indexed=0；Chroma 不可用时保留队列且不推进游标。同步使用同 ID，更新前清除旧元数据，避免旧 paper 标志残留。读取时指纹不匹配的旧版本被拒绝使用；无 SQLite 原文的孤立索引不能进入上下文。40 条事件的分批测试覆盖“其他记录可召回但部分 ID 丢失”；真实 Chroma 测试覆盖删除其中一个向量及原文论文范围变更后的恢复。

### 实际执行

| 命令/范围 | 结果 | 独立原始输出 |
|---|---|---|
| MemoryManager + 新增召回 + Reliability | 42 passed，0 failed，0 skipped | `memory-fixes/targeted-v2.xml` |
| 补充 l2/cosine、无命中不误恢复断言 | 14 passed，0 failed，0 skipped | `memory-fixes/recall-v3.xml` |
| 最终完整 pytest | **93 passed，0 failed，0 skipped**；9.68 秒；68 条 Pydantic 弃用警告 | `memory-fixes/final-tests.xml` |
| 源码、测试、集成脚本 compileall | 通过 | `memory-fixes/verification.json` |
| git diff --check | 通过 | `memory-fixes/verification.json` |

所有测试在原 `.venv-gpu` 环境执行。真实 Chroma 使用临时目录和确定性单位向量，验证真实存储 API/距离/恢复，不是 BGE 语义质量评测。Redis 的既有故障/恢复协议测试通过，本轮未启动 Redis 服务、未调用 OpenCode、未重新下载权重，也未改变 FAISS 检索或正式数据。此前 79 项/六轮集成属于下方历史验收。

```powershell
.venv-gpu/Scripts/python.exe -m pytest tests/test_memory_manager.py tests/test_memory_recall_fixes.py tests/test_reliability.py -q
.venv-gpu/Scripts/python.exe -m pytest -q --junitxml=reports/memory-fixes/reproduced.xml
.venv-gpu/Scripts/python.exe -m compileall -q src/paper_agent tests scripts/check_memory_integration.py
git diff --check
```

本轮没有剩余阻塞测试。已知限制：相关性阈值和显式事件标记会漏召回/漏保存部分自然表达；超过一个批次的缺失是渐进恢复，不是即时全量修复；原文候选装载、归档读取与完全重复检查仍为本地规模设计。历史 Chroma 1.x Windows 持久化兼容问题、Pydantic 弃用警告和通用语义事实裁决不在本轮修复范围内。新增依赖为零。

## 2026-10-25 初始升级验收（历史记录）

日期：2026-10-25。基线提交 `4368a62a4ba15160fb5ff452194fd10bd2737c4e`，本轮为未提交工作区修改；已有未提交的评测审计工作全部保留。本报告不替代此前 LLM/检索评测报告。

## 实际改动

- `src/paper_agent/memory_stores.py`：SQLite 事实版本/事务/软删除、会话快照、事件原文与待同步标志；Redis TTL 工作缓存；Chroma 显式 BGE 向量、模型签名、session/paper/event 元数据过滤。
- `src/paper_agent/memory_manager.py`：统一读取、候选提取、冲突与更新、事件选择/去重、召回预算、原有压缩调用和跨存储补偿。新增机制没有调用 Mem0。
- `src/paper_agent/pipeline.py`：可注入 manager，或通过 `MEMORY_BACKEND=three_layer` 启用；NLU 前后读取、Memory 阶段提交，保留七阶段和 legacy 行为。
- `tests/test_memory_manager.py`：11 项针对性测试；`scripts/check_memory_integration.py`：六轮真实存储/BGE/FAISS 集成检查，固定回答器，不调用 LLM。
- `.env.example`、`pyproject.toml`、`constraints-memory.txt`：必要依赖、配置与原 GPU 环境传递依赖约束。
- `README.md`、`docs/README.md`、`docs/architecture.md`、`docs/memory.md`、`docs/runtime-tracing.md`：实际调用链、启动、迁移、证据与恢复边界。

本轮没有编辑论文检索算法、评测数据或已有评测结果。`retrieval.py` 中原有的 4 行未提交注释保持原样，不能计为本轮修改。

## 调用链与行为

`run → load_working_memory` 从 SQLite 确认已提交版本，检查 Redis 并恢复缓存/文件投影、待同步事件。过期缓存不会丢失尚存于 SQLite 的状态。NLU 前使用当前问题、后使用 standalone_query 召回事实及情景，最多 6 条/5000 字符，保留旧 ContextArchive 读取。

Memory 阶段提取候选，SQLite 事务提交事实、事件原文、状态和 turn 账本；压缩复用现有函数，归档先于裁剪。事务成功后才更新 Redis/Chroma/文件。后续同步失败保留 committed + degraded；再次读取/提交补偿。相同 turn_id/问题重放已有结果，不重复消息、事实或事件。旧版本并发提交明确失败。

同 canonical_key/内容忽略重复；明确替代当前 fact_id 才 supersede 旧版本；未明确修正或不同 key 的高词项重合内容保留 conflict。用户 `更新记忆:键=值` 是显式替代入口。源类型独立，普通用户偏好无需论文引用。旧文件事实首次导入为 legacy_unverified conflict，不能自动成为已验证论文事实。

论文候选必须通过本轮 `validate_citations`。仅完整句子/全文逐字摘录通过保守支持规则成为 active；其他合法引用的生成改写保留 semantic_support_unverified conflict。历史情景不作为论文证据，Generate 仍使用本轮真实 chunks。本轮没有把同模型判断包装成人工语义验证。

## 实际执行结果

| 验证 | 结果 | 原始记录 |
|---|---|---|
| 修改前原测试 | 68 passed | 会话命令输出 |
| 最终完整测试 | **79 passed，0 failed，0 skipped**，7.66 秒 | `memory-upgrade/final-recovery-tests.xml` |
| 真实存储 + BGE/FAISS 六轮集成 | **passed**，21.14 秒 | `memory-upgrade/integration-verified/result.json` 和 `traces/` |
| Redis 实际 TTL 过期、恢复及重放 | passed，恢复后仍为 6 turns、6 episodes | 同上 |
| Chroma 会话/论文/event_type 过滤 | passed | 同上及真实 Chroma 单元测试 |
| 依赖一致性 | `No broken requirements found` | 实际运行 `pip check` |
| 补丁格式 | `git diff --check` 通过 | 实际命令输出 |

真实集成使用 Redis Server 7.0.15（隔离 WSL 实例端口 16379）、redis-py 6.4.0、Chroma 0.6.3、FlagEmbedding 1.3.5、faiss-cpu 1.15.1；复用 `models/bge-m3`、`indexes/bge`，设置模型离线模式，没有重新下载权重或重建论文索引。语料哈希：`03fd776bc579c2e7137ef29f3903a8b0955120b6ffe6e227e682d92623756f0e`。六轮产生 6 条情景、1 条重复去重后的有效原文事实；压缩后活动消息 8 条，所有六轮 trace 无 degraded。

单元测试中的 Redis/故障对象为协议替身，实际 TTL 和网络恢复由独立集成脚本验证。完整测试包含原引用、Pipeline、检索、评测与可靠性测试。本轮没有昂贵的 OpenCode 评测，API 调用数为 0；固定回答器的通过率不是模型答案准确率。

## 发现的问题与保留的失败

1. pip 缓存写入阻塞：有限诊断定位到缓存临时文件；使用 `--no-cache-dir` 解决。早期自动审批额度阻断已解除。
2. Redis 旧实例未运行：首次 `integration-full/result.json` 保存 TimeoutError；重启隔离实例后真实连接通过。
3. Chroma 1.5.9、1.3.7 原生持久化在此 Windows 环境均报 SQLite code 14。目录预创建、绝对/相对路径和空库预创建未解决；0.6.3 Python 后端通过实际测试，因此固定 0.6.3。没有改用假向量库通过验收。失败 JUnit 保存在 `unit-tests.xml`、`targeted-tests*.xml`、`targeted-compatible.xml`。
4. 首次 BGE 集成发现完整多句原文被误判为未验证：修复规则同时检查完整 chunk 与完整句子，补充回归断言。失败结果保存于 `integration-retry/`，修复后结果为 `integration-final/` 和最终 `integration-verified/`。
5. 索引整体丢失：新增 SQLite 原文降级读取及重建排队测试。部分索引缺失仍需人工按 SQLite 原文重建（见下述边界）。
6. 新依赖最初与原环境 protobuf/rich/OpenTelemetry 冲突：已通过 constraints 固定兼容范围；最终 pip check 无冲突。安装临时 wheel 目录已清理。

## 复现与启动

在原工作目录执行；命令保留既有语料、模型、索引和 `.env`。

```powershell
.venv-gpu/Scripts/python.exe -m pip install --no-cache-dir -c constraints-memory.txt -e '.[memory]'
.venv-gpu/Scripts/python.exe -m pip check
.venv-gpu/Scripts/python.exe -m pytest -q --junitxml=reports/memory-upgrade/reproduced-tests.xml
docker run -d --name paper-agent-redis -p 127.0.0.1:6379:6379 redis:7-alpine
.venv-gpu/Scripts/python.exe scripts/check_memory_integration.py --redis-url redis://localhost:6379/0 --output reports/memory-upgrade/reproduced-integration
$env:MEMORY_BACKEND='three_layer'
$env:REDIS_URL='redis://localhost:6379/0'
.venv-gpu/Scripts/python.exe -m paper_agent.cli chat --index indexes/bge --mode hybrid --encoder models/bge-m3 --device cuda:0 --session memory-demo
```

已有 Redis 时跳过 Docker 启动并使用其地址；集成输出目录要求不存在，以避免覆盖历史结果。本轮隔离 Redis 在验收后关闭，不依赖后台服务留存。默认 legacy 不强制用户安装存储依赖；三层模式须显式启用。

## 限制

- Chroma 0.6.3 在当前 Pydantic 版本下产生 21 条弃用警告，未删除或屏蔽；1.x Windows 原生后端兼容问题尚未解决。其 FastAPI/HNSW 包是 Chroma 的传递依赖，本项目没有新增 Web 服务，也没有改变 FAISS 检索结构。
- 语义改写不能自动成为有效论文事实；相似冲突只使用保守规则，不是通用语义裁决。语义压缩仍为现有 LLM 辅助摘要，不是独立事实认证。
- Chroma/Redis 是可恢复投影，不存在跨库原子事务；恢复每次最多重试 32 个待同步事件。部分丢失/损坏的 Chroma 索引可用新 collection，然后逐项执行 `manager.episodic.write(episode)`（episode 来自 `manager.facts.episodes(session_id)`）重建；不可删除 SQLite 原文。
- 工作区全文事实和账本长期规模、归档线性扫描、SQLite 单写者吞吐仍适合本地规模。5000 字符是上下文上限而非模型 token 计量。
- CLI bm25 模式没有 encoder，三层模式要求已有 dense/hybrid/rerank 编码器；不自动下载替代模型。
- 本次完成了存储和编排验收，没有重新宣称已有论文问答算法增益，也没有新增人工金标或真实 LLM 质量评测。
