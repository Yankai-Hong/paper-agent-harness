# 工具循环与执行追踪

[runtime.py](../src/paper_agent/runtime.py) 改编自固定版本 learn-claude-code s02：提交 messages，接收 tool_use，执行允许的工具，为每个调用 ID 写回 tool_result，直到完成或达到上限。新增 JSON schema 检查、工具白名单、错误反馈、hooks 和 usage 汇总。MIT 来源见 [许可证说明](../THIRD_PARTY_NOTICES.md)。

七阶段主流程使用显式领域编排；通用工具循环作为可测试的基座保留，不通过 shell 执行论文中的内容。API 适配器采用 Chat Completions 的 JSON 对象输出，不依赖供应商原生工具调用协议。

## Trace

三层模式额外记录 `memory_events`：长期声明 active/conflict/update/duplicate、事件 queued/skip、Redis/Chroma 故障类型、补偿方式。服务错误只记录类型，不包含连接 URL 或密码。SQLite 提交后索引/缓存同步失败是 `commit_status=committed` 且 `degraded=true`；提交前失败仍为 uncommitted。该模式使用版本检查拒绝同 session 的过期并发提交。

每轮一个 turn_id，按阶段记录 input、output、attempt、status 和 elapsed_ms。二次检索分别留下 Retrieve/Rerank/Reflect span。发生异常也写入 trace；未经过的阶段标为 skipped，而不是伪造成功。

交互 trace 位于 `runs/traces/<session>/`，批量 trace 位于指定输出目录下。`/trace` 展示当前进程最近成功一轮的完整 trace；失败轮直接检查目录里的最新文件。`/corpus` 显示索引元信息，`/memory` 展示工作记忆状态。

API 记录包含请求/返回模型名、耗时、状态、原始 usage 和归一化 usage。OpenAI-compatible 的 prompt_tokens 已包含缓存命中，因此归一化为非缓存 input_tokens 加 cache_read_input_tokens，不能再重复加缓存数。上游 Anthropic 风格计数在通用 runtime 中另行处理。

API 缺少 usage 时 usage_complete=false；不能把未知 token 当作真实零成本。HTTP/格式错误也计入请求次数上限；Provider 有有限重试，Pipeline 有显式降级，详见下文。trace 不保存密钥/Authorization；会保存论文问题、证据和答案，应按自己的数据权限管理运行目录。

## 本轮扩展

稳定 turn_id 可由调用方传入；trace 增加 archive_recall、degraded/degradation_reasons、commit_status。API 每次尝试记录 finish_reason、reasoning_tokens、repair、retry_reason、retry_delay_seconds、potential_duplicate_cost。缺 usage 为未知成本，失败和重试不得从分母删除。状态提交幂等；不支持同 session 并发提交。


## 修订追踪与遥测

本地 Turn → Stage invocation_id → 每次实际请求 request_id 独立关联，记录开始/结束时间、case_id/run_id、provider 实际 usage、Schema修复与重试。没有 usage 时保持 unknown。eval_tasks 的规则分数和 eval_cli 的同模型 Judge 分数关联回本地 trace，unknown 不换算为0；可观测成功不代表质量验收。

PAPER_TELEMETRY_ENABLED 默认 false；显式开启才加载固定 Langfuse SDK 4.8.1（Python 3.10）并导出。LANGFUSE_BASE_URL/PUBLIC_KEY/SECRET_KEY 从环境读取；不升级现有环境。白名单仅元数据，标识哈希，字符串最多120字符，不上传正文、问题、完整偏好或任意异常文本。散列标识是可关联的伪匿名数据，低熵 ID 可能被猜测，不是匿名化。容量1后台队列，有限 flush，失败不改变业务提交。外部 Span 是异步投影，真实业务起止时间和耗时在 metadata 中，SDK 瀑布图原生时间是投影时间。真实服务已读回关键字段，完整树和界面仍待验收。

追踪标识采用 `paper-agent-trace-v2`：通过规范化编码的 `(命名空间, run_id, session_id, turn_id)` 生成确定性 32 位十六进制 ID。无 run_id 的普通交互使用独立的 interactive 命名空间和 session_id、turn_id；同轮重复导出稳定，不同实验 run 隔离。Trace 本地 `observability_mapping` 记录映射与算法版本；Score ID 再结合 Trace ID、来源和名称计算，补分复用同一 ID。带 run_id 的本地 Trace 位于 `traces/runs/<run 哈希>/<turn_id>.json`，无 run_id 沿用旧路径；调用方应读取返回的 `trace_path`。旧 Trace 若无可核对映射，补分只保存在本地并标记 withheld，不猜测远端归属。

允许导出的 RAG/记忆摘要仅包括候选与重排数量、证据 ID 的哈希摘要、子目标覆盖计数、记忆召回/修复数量及白名单降级原因；不输出问题、论文正文、偏好、密钥或异常正文。覆盖只表示候选存在，不表示语义支持。Langfuse 的 [Trace ID 文档](https://langfuse.com/docs/observability/features/trace-ids-and-distributed-tracing) 说明 32 位十六进制 ID 和确定性映射；本实现的具体映射为项目自定义版本。

SDK 不支持自定义 observation ID，因此相同 Trace ID 不能独自防止重复子树。适配器在创建树之前，在 `.cache/paper-agent-telemetry`（可通过 `PAPER_TELEMETRY_DEDUP_DIR` 指定）对目标实例/项目与 Trace ID 创建原子占位标记；共享标记目录的进程对同一目标最多尝试一次建树。该策略不等同于服务端全局精确一次：进程或网络在发送期间失败时，远端树可能缺失或不完整且不会自动重发；不同机器或标记目录不共享状态。标记不保存凭据，需与本地 Trace 一并妥善保留，不能根据文件存在推断服务端接收成功。Score 通过稳定的 Score ID 独立更新。相关测试使用替身，服务端读回尚未验收。

适配器的发送状态使用 `tree_sent_unverified`、`tree_already_claimed_unverified`、`scores_sent_unverified` 等口径：即使本地 SDK/flush 返回，也不能推断服务端完整接收；占位后重放跳过建树时更不能报为已送达。真实读回至少需要可访问且兼容的 Langfuse 实例、Python 3.10 可用的 SDK 4.8.1、实例地址/公钥/密钥和明确的数据上传授权。

2026-10-30 用户授权固定合成样本的脱敏元数据上传，配置实例后已实际上传并从服务端读回关键字段；Turn 外部父节点和界面呈现仍待核实。最新状态见[真实服务读回记录](../reports/revision/LANGFUSE-LIVE-2026-10-30.md)，此前未配置实例时的[准备报告](../reports/revision/LANGFUSE-2026-10-30.md)保留历史状态。

`PAPER_TELEMETRY_FLUSH_SECONDS` 默认为 0.5，范围 0–2；仅限制业务线程的本地等待，队列满、超时、短进程退出或异常退出均可能丢失投影。分数采用稳定 ID 更新；unknown 不发送，如果之前已发送已知值而本地后来变为 unknown，远端旧分数可能残留。请以本地逐例 Trace 和评分为准。

适配器固定 SDK 4.8.1，并使用独立的 OpenTelemetry provider 和本适配器登记的 span ID 白名单，避免进程内其他追踪数据被一并导出；已由其他组件使用相同公钥初始化的不兼容 SDK 资源会令导出失败而不影响业务。它依赖该 SDK 版本的内部资源及退出机制，不能未经验证地换版本。SDK 的协议必需认证头及固定环境/版本元数据不属于业务正文；SDK 4.8.1 的分数采样还可能受进程全局 OTel sampler 影响。服务端读回才能确认实际投影。

## 安装与授权后的服务端读回

可以使用已有 Langfuse Cloud 或已有自托管实例；无需为了此项目新建云账户。官方 [兼容矩阵](https://langfuse.com/docs/compatibility) 指出 SDK v5 要 Python 3.11，现有 Python 3.10 保持 SDK v4；自托管 v3 需至少 3.63.0，v4 查询接口依赖服务端版本，部署前核对矩阵。以下针对服务端 v4 和 SDK 4.8.1 的 Scores v3 读回；如是 v3 服务，需按官方兼容矩阵和 legacy API 调整，不能把 v4 读回命令当作已验证。

为避免改动现有虚拟环境，可在一个**新的**独立目录安装 SDK 及依赖，先核对依赖冲突；仅在本次进程引用该目录（不要覆盖 `.env`，不要将密钥放入命令历史或报告）：

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'
& $py -m pip install --target .cache/langfuse-runtime 'langfuse==4.8.1'
$env:PYTHONPATH = (Resolve-Path .cache/langfuse-runtime).Path + ';' + (Resolve-Path src).Path
& $py -c "from importlib.metadata import version; import langfuse; print(version('langfuse'))"
& $py scripts/check_langfuse_sdk_offline.py
```

先通过安全渠道配置 `LANGFUSE_BASE_URL`、`LANGFUSE_PUBLIC_KEY` 和 `LANGFUSE_SECRET_KEY` 到本进程环境，并明确允许**合成样本元数据**上传。分别设置 `PAPER_TELEMETRY_ENABLED=true`、`PAPER_TELEMETRY_SAMPLE_RATE=1`、`PAPER_TELEMETRY_FLUSH_SECONDS=2`。运行模型问答须另有现成模型配置和使用授权；合成样本与两次新运行的命令及预算见 [评测说明](#合成样本和规则评分验证)。客户端标识和哈希不等于匿名，不要使用旧的个人会话。

从两个新运行的本地 Trace 各读取 `observability_mapping.langfuse_trace_id`；按 [官方查询文档](https://langfuse.com/docs/api-and-data-platform/features/query-via-sdk) 在授权环境使用 SDK 的 `api.observations.get_many(trace_id=..., fields='core,basic,metadata,model,usage')`，按 `parent_observation_id` 核对 Turn → Stage → LLM request、阶段状态/实际耗时 metadata、模型与 provider usage、检索及记忆摘要。用 `api.scores_v3.get_many_v3(id=...)` 查分数；ID 可由本地 Trace ID 和 `paper_agent.observability.score_id(trace_id, 'rule', 'execution_success')` 确定，核对 `trace_id`、值和来源。比较两次 run 的 Trace/Score ID，在同一共享标记目录下重放并检查无新增树；对照本地 Trace 和逐例结果。新数据通常需 15–30 秒可查，设有限截止时间轮询；缺失、不完整、缺 usage 或读回错误均不能写通过。[官方兼容说明](https://langfuse.com/docs/compatibility) 和服务端/API 版本需随运行记录。真正读回需同时完成这些检查，SDK 初始化、flush 或测试替身不是验收。

也可在服务端 v4 和 SDK 4.8.1 下，对两组新输出中的本地 Trace 使用只读核验入口：

```powershell
& $py scripts/check_langfuse_readback.py $traceA $traceB
```

先将 `$traceA`、`$traceB` 分别设为两组逐例结果的 `trace_path`，不要把凭据写在命令行。脚本按两组本地身份读取并核对服务端结果，限制等待和分页，不上传、调用模型或打印响应正文；无匹配记录及缺字段不会判通过。它不自动轮询新数据的入库延迟：若结果尚未入库，等待后重新运行。脚本不核验重放或界面，需另行检查并保留证据；它自身的成功不能代表完整端到端验收。

### 合成样本和规则评分验证

以下先离线建立固定一论文/一问题数据，拒绝覆盖已有目录；之后两条命令调用实际模型，最多各 3 次、合计 6 次，每次输出上限 1024 token，关闭重试和修复，45 秒/请求。请求失败、实际 token 和 unknown 都要记录；请先配置模型、实例和上传授权。目录已有时换新名字，评测 CLI 自身可能覆盖指定输出。

```powershell
& $py scripts/prepare_langfuse_fixture.py --output reports/revision/langfuse-fixture-new
$env:MEMORY_BACKEND='sqlite_local'; $env:PAPER_DECOMPOSITION='false'
$env:API_MAX_RETRIES='0'; $env:API_REPAIR_RETRIES='0'; $env:API_TIMEOUT='45'
$env:PAPER_TURN_MODEL_REQUESTS='3'
foreach ($run in @('reports/revision/langfuse-check-a','reports/revision/langfuse-check-b')) {
  if (Test-Path $run) { throw "Output already exists: $run" }
  & $py -m paper_agent.cli eval e2e reports/revision/langfuse-fixture-new/cases.jsonl --index reports/revision/langfuse-fixture-new/index --mode bm25 --output $run --max-calls 3 --max-tokens 1024 --max-retrieval-attempts 1
}
```

输出中的 `predictions.jsonl`、`metrics.json` 与本地 Trace 保留实际结果、配置哈希和 provider usage；不要覆盖旧运行。关闭遥测可在本进程设置 `PAPER_TELEMETRY_ENABLED=false`。

预算默认8实际模型请求、6检索调用、90聚合候选、60重排、180秒；全部重试与修复计数。时间为协作式 deadline，不能强制中断 native GPU/DNS，HTTP timeout 受剩余时间限制。API_CIRCUIT_ENABLED 默认 false；启用后按 provider/model 在进程内共享 CLOSED/OPEN/HALF_OPEN，阈值 API_CIRCUIT_THRESHOLD=3、冷却 API_CIRCUIT_COOLDOWN=30。半开只允许一个探测；认证、限流、服务端、连接、超时与 schema/budget 分开，正常拒答不计健康失败。未配置且未验证独立备用模型，全部失败明确终止，不宣称主备高可用。
