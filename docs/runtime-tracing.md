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

PAPER_TELEMETRY_ENABLED 默认 false；显式开启才加载 Langfuse SDK v4（Python 3.10）并导出。LANGFUSE_BASE_URL/PUBLIC_KEY/SECRET_KEY 从环境读取；不升级现有环境。白名单仅元数据，标识哈希，字符串最多120字符，不上传正文、问题、完整偏好或任意异常文本。容量1后台队列，有限 flush，失败不改变业务提交。外部 Span 是异步投影，原始起止时间在 metadata 中。本轮没有真实服务读回验收。

追踪标识采用 `paper-agent-trace-v2`：通过规范化编码的 `(命名空间, run_id, session_id, turn_id)` 生成确定性 32 位十六进制 ID。无 run_id 的普通交互使用独立的 interactive 命名空间和 session_id、turn_id；同轮重复导出稳定，不同实验 run 隔离。Trace 本地 `observability_mapping` 记录映射与算法版本；Score ID 再结合 Trace ID、来源和名称计算，补分复用同一 ID。带 run_id 的本地 Trace 位于 `traces/runs/<run 哈希>/<turn_id>.json`，无 run_id 沿用旧路径；调用方应读取返回的 `trace_path`。旧 Trace 若无可核对映射，补分只保存在本地并标记 withheld，不猜测远端归属。

允许导出的 RAG/记忆摘要仅包括候选与重排数量、证据 ID 的哈希摘要、子目标覆盖计数、记忆召回/修复数量及白名单降级原因；不输出问题、论文正文、偏好、密钥或异常正文。覆盖只表示候选存在，不表示语义支持。Langfuse 的 [Trace ID 文档](https://langfuse.com/docs/observability/features/trace-ids-and-distributed-tracing) 说明 32 位十六进制 ID 和确定性映射；本实现的具体映射为项目自定义版本。

SDK 不支持自定义 observation ID，因此相同 Trace ID 不能独自防止重复子树。适配器在创建树之前，在 `.cache/paper-agent-telemetry`（可通过 `PAPER_TELEMETRY_DEDUP_DIR` 指定）对目标实例/项目与 Trace ID 创建原子占位标记；共享标记目录的进程对同一目标最多尝试一次建树。该策略不等同于服务端全局精确一次：进程或网络在发送期间失败时，远端树可能缺失或不完整且不会自动重发；不同机器或标记目录不共享状态。标记不保存凭据，需与本地 Trace 一并妥善保留，不能根据文件存在推断服务端接收成功。Score 通过稳定的 Score ID 独立更新。相关测试使用替身，服务端读回尚未验收。

适配器的发送状态使用 `tree_sent_unverified`、`tree_already_claimed_unverified`、`scores_sent_unverified` 等口径：即使本地 SDK/flush 返回，也不能推断服务端完整接收；占位后重放跳过建树时更不能报为已送达。真实读回至少需要可访问且兼容的 Langfuse 实例、Python 3.10 可用的 SDK v4、实例地址/公钥/密钥和明确的数据上传授权；当前缺少这些条件，未上传。

预算默认8实际模型请求、6检索调用、90聚合候选、60重排、180秒；全部重试与修复计数。时间为协作式 deadline，不能强制中断 native GPU/DNS，HTTP timeout 受剩余时间限制。API_CIRCUIT_ENABLED 默认 false；启用后按 provider/model 在进程内共享 CLOSED/OPEN/HALF_OPEN，阈值 API_CIRCUIT_THRESHOLD=3、冷却 API_CIRCUIT_COOLDOWN=30。半开只允许一个探测；认证、限流、服务端、连接、超时与 schema/budget 分开，正常拒答不计健康失败。未配置且未验证独立备用模型，全部失败明确终止，不宣称主备高可用。
