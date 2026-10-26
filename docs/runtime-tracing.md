# 工具循环与执行追踪

[runtime.py](../src/paper_agent/runtime.py) 改编自固定版本 learn-claude-code s02：提交 messages，接收 tool_use，执行允许的工具，为每个调用 ID 写回 tool_result，直到完成或达到上限。新增 JSON schema 检查、工具白名单、错误反馈、hooks 和 usage 汇总。MIT 来源见 [许可证说明](../THIRD_PARTY_NOTICES.md)。

七阶段主流程使用显式领域编排；通用工具循环作为可测试的基座保留，不通过 shell 执行论文中的内容。API 适配器采用 Chat Completions 的 JSON 对象输出，不依赖供应商原生工具调用协议。

## Trace

三层模式额外记录 `memory_events`：事实 active/conflict/update/duplicate、事件 queued/skip、Redis/Chroma 故障类型、补偿方式。服务错误只记录类型，不包含连接 URL 或密码。SQLite 提交后索引/缓存同步失败是 `commit_status=committed` 且 `degraded=true`；提交前失败仍为 uncommitted。该模式使用版本检查拒绝同 session 的过期并发提交；本文原文件模式的并发限制仍适用于 legacy。

每轮一个 turn_id，按阶段记录 input、output、attempt、status 和 elapsed_ms。二次检索分别留下 Retrieve/Rerank/Reflect span。发生异常也写入 trace；未经过的阶段标为 skipped，而不是伪造成功。

交互 trace 位于 `runs/traces/<session>/`，批量 trace 位于指定输出目录下。`/trace` 展示当前进程最近成功一轮的完整 trace；失败轮直接检查目录里的最新文件。`/corpus` 显示索引元信息，`/memory` 展示当前会话事实。

API 记录包含请求/返回模型名、耗时、状态、原始 usage 和归一化 usage。OpenAI-compatible 的 prompt_tokens 已包含缓存命中，因此归一化为非缓存 input_tokens 加 cache_read_input_tokens，不能再重复加缓存数。上游 Anthropic 风格计数在通用 runtime 中另行处理。

API 缺少 usage 时 usage_complete=false；不能把未知 token 当作真实零成本。HTTP/格式错误也计入请求次数上限；Provider 有有限重试，Pipeline 有显式降级，详见下文。trace 不保存密钥/Authorization；会保存论文问题、证据和答案，应按自己的数据权限管理运行目录。

## 本轮扩展

稳定 turn_id 可由调用方传入；trace 增加 archive_recall、degraded/degradation_reasons、commit_status。API 每次尝试记录 finish_reason、reasoning_tokens、repair、retry_reason、retry_delay_seconds、potential_duplicate_cost。缺 usage 为未知成本，失败和重试不得从分母删除。状态提交幂等；不支持同 session 并发提交。
