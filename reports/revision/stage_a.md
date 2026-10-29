# A：可选可观测

本地 Trace 仍为权威；Turn→Stage invocation→实际请求分别生成关联 ID、起止时间。修复和传输重试保持独立请求记录，usage 只使用 provider 实际返回值。case_id/run_id 为调用方可传字段。

新增 observability.py，默认 PAPER_TELEMETRY_ENABLED=false。Exporter 白名单元数据、标识哈希、120 字符截断，不导出问题、偏好、正文、异常内容或密钥。单 daemon、容量1队列、有限等待；不可达或阻塞会丢弃外部投影，不影响 SQLite 提交。外部 span 是异步投影，原始时间在 metadata 中，以本地起止时间为准。E2E 规则分数和可选同模型 Judge 分数关联本地 trace；unknown 不伪造为零，外部分数是独立投影。后者未进行真实 Langfuse 读回验证。

官方资料核对（2026-10-28）：https://langfuse.com/docs/compatibility 与 https://langfuse.com/docs/observability/sdk/upgrade-path/python-v4-to-v5 。Python 3.10 保持 SDK v4；v5 要求 Python 3.11 和服务 v4.59.0+。v4 对自建 v3 的版本要求须对实际实例核验。适配器延迟导入且检查 v4，未安装升级 SDK；配置 LANGFUSE_BASE_URL/PUBLIC_KEY/SECRET_KEY 仅从环境读取。

验证：tests/test_revision_observability.py 覆盖关闭、重试树、脱敏、不可达、阻塞有限 flush、异常退出及 SDK 替身。最终全量 151 passed。真实服务无读回验收，不宣称上传成功或模型效果改善。开关关闭为回滚方式，无数据库变化。
