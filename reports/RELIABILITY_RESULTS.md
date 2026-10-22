# 可靠性与恢复测试

本轮在原 `.venv-gpu` 环境运行，最新结果：{'tests': 61, 'failures': 0, 'errors': 0, 'skipped': 0}。机器可读记录为 acceptance-tests.xml；阶段记录 baseline-tests.xml、phase-I-tests.xml、phase-II-III-tests.xml、phase-IV-tests.xml、phase-V-preflight-tests.xml 保留。

这些是离线/模拟 HTTP 故障注入，不能代替真实 API 效果指标：

| 故障 | 验证行为 |
|---|---|
| 429、503、网络 timeout | 有限重试、耗时/原因/未知成本记录 |
| 400、401、403 | 不重试，明确失败 |
| 非法 JSON、Schema 不匹配、非法 Reflect 决策 | 最多一次结构修复；仍失败则 fail closed |
| 连接预算短于响应头等待 | 建连后使用总请求预算；总超时仍失败 |
| 重排器缺失/不可用 | 保留检索排序，Trace degraded |
| hybrid 单路异常 | 配置允许时尝试另一通道；不能恢复则失败 |
| 错误引用 | 受控拒答，不写事实，Trace degraded |
| 语义摘要失败 | 确定性裁剪及完整归档 |
| 归档写入失败 | 回滚本轮消息、事实、计数，Trace error |
| 重复 turn_id、进程恢复 | 不重复 API、消息、事实或提交；相同 ID 不同问题拒绝 |
| 超阈值长期会话 | 早期消息离开近期窗口仍可召回；论文引用仍指向 chunk |
| 更新冲突与跨 session | 旧事实 superseded、当前事实优先、不同 session 不返回记录 |

修复过程中测试实际揭示并修正：401 被误重试、旧 SessionMemory.load 未识别新原子快照、第二次归档查询覆盖初次高相关结果。t3 真实预算探针还发现早期连接超时错误约束响应头等待，旧失败保留，修正后真实调用通过。

边界：没有完整阶段 Checkpoint Resume；只有请求级有限重试与幂等提交。单 session 不支持多进程并发写；同一进程调用方应串行执行。真实线上断网/真实 429 并未主动制造，相关机制证据为故障注入。磁盘原子替换不是跨机器事务。
