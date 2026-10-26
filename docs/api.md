# OpenCode API 与可靠性

Phase A 最新调用成本见 `reports/phase-a/cost-ledger.json`。部分 Judge 请求实测 elapsed 约 8,895 秒，远超配置 120 秒后最终记录传输失败并重试；服务或宿主暂停原因尚未确认。保留这些异常，不将此批次壁钟耗时当作正常吞吐或硬实时超时保证。活跃进程的超时故障注入测试与该运行观测分别报告。

本轮直接使用原 `.env` 的 `https://opencode.ai/zen/go/v1` 和 `deepseek-v4.1-flash`，未替换 provider 或凭证。`APIConfig.from_env` 使用进程环境优先；CompatibleReasoner 为每次请求注入稳定的 `x-opencode-session`。请求追加 `/chat/completions`，发送 JSON mode 并在本地验证 Schema。

本次 t3 固定输入复测：2048 输出预算 finish_reason=length、reasoning_tokens=2048；4096 和 8192 成功。4096 加 thinking=disabled 的响应 reasoning_tokens=0，说明本次参数实际生效，不保证其他模型或未来服务行为。详细原始 usage 见 `reports/acceptance/budget-v2/results.json`。正式开发批次用 CLI 显式覆盖 4096/disabled，原 .env 不改。

| 配置 | 默认 | 作用 |
|---|---:|---|
| API_MAX_TOKENS | 2048 | 完成预算（可能含推理）；eval 可用 --max-tokens 覆盖 |
| API_THINKING | unset | enabled/disabled；eval 可用 --thinking 覆盖 |
| API_CONNECT_TIMEOUT | 15 s | 建连超时 |
| API_TIMEOUT | 120 s | 单次 HTTP 请求总预算，含读取正文 |
| API_MAX_RETRIES | 2 | 429、5xx、网络异常的额外重试次数 |
| API_REPAIR_RETRIES | 1 | 非法 JSON/Schema/截断后最多一次结构修复 |
| API_RETRY_BASE_SECONDS | 1 s | 指数退避基数，加随机抖动 |
| API_MAX_CALLS | 100 | 每个 reasoner 实例的实际请求上限，重试计入 |
| API_MAX_INPUT_CHARS | 80000 | 输入字符硬上限，不是精确 tokenizer 预算 |

400/401/403 等参数或鉴权错误不重试。修复请求要求更短且完整的 JSON，不伪造路由。总逻辑调用最多初次+2次传输重试+1次修复（还受 max_calls 限制），每次 HTTP 分别受 timeout 限制。HTTP 超时可能已在服务端执行，Trace 标记 potential_duplicate_cost，未知用量不能当零成本。初次旧超时实现的失败记录保留于 budget/，修正后记录为 budget-v2/。

只保存错误类型、HTTP 状态、请求哈希、模型、finish_reason、reasoning_tokens、usage、耗时与重试原因，不存 Authorization/key 或供应商错误正文。prompt_tokens 已含 cached_tokens，规范化后用非缓存 input+cache_read+output 汇总一次。无 usage 标为未知。

多个评测进程/会话的上限独立，max_calls 不是整个账户预算，也不能读取账户剩余额度。批量实际调用总量从保存的所有 Trace 汇总。遇到额度或服务异常仅有限重试，保留失败分母。真实 API 与模拟 HTTP/故障注入测试分别见 LIVE_EVAL_RESULTS 和 RELIABILITY_RESULTS。
