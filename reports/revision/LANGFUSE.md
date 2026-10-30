# Langfuse 可视化阶段报告

记录日期：2026-10-29。保留此前报告和未确认旧数据；该轮执行时未提交或推送。公开的测试日志已将本机用户名路径替换为 `<USER>` 并清理行尾空格，测试结果未改动。Git HEAD `a5754aa40eede0df1e6a556c128386647f9a5e4d` 是本轮开始的基线，不代表含未提交修改的源码。最终 `src/paper_agent/*.py` 各文件 SHA-256 和其他脚本哈希见 [源码冻结](langfuse-source-freeze.json)；排序键、紧凑 JSON 的源码集哈希 `a73895712b087f39d241ff1c14382798bacbac246f3ecc449d913d9094fa7b27`。

## 状态和范围

| 项目 | 状态 | 证据与限制 |
|---|---|---|
| 本地实现 | 已完成 | 复用 TurnTrace、阶段调用、Provider 请求和现有评测规则；本地 JSON 是权威记录。 |
| SDK 和故障替身 | 通过 | 白名单、缺 SDK、不可达/超时替身、队列满、有限等待、异常和短进程退出、SQLite 提交及重放、评分隔离和 unknown。不是远端验收。 |
| 真实 SDK 离线 | 通过，有边界 | Python 3.10.20、SDK 4.8.1、OpenTelemetry 1.37.0；网络禁用、内存 exporter；两个 run 共 2 Trace/6 Span、4 条截获 Score 事件，见 [日志](langfuse-sdk-offline-final.log)。虚拟父 span 的真实服务端呈现、服务端接收及 Score 归属尚未知。 |
| 真实服务读回 | **未验收** | 未发现完整服务地址和凭据，也未取得合成测试数据上传的明确授权；服务端版本未知，未发起上传或创建项目。 |
| 真实模型 | 本轮未调用 | 模型请求 0、实际 token 不适用、云端上传 0；没有新的质量结论。 |

本地两个现有虚拟环境分别为 Python 3.10.20 和 3.10.11，均未安装 SDK。为离线检查，将 SDK 4.8.1 和与已有 1.37.0 匹配的 OTLP HTTP exporter 放入独立 `.cache/langfuse-sdk-v4`；没有修改虚拟环境、Python、CUDA、模型或 `.env`。读取 [官方版本兼容](https://langfuse.com/docs/compatibility) 和 [v4 到 v5 迁移说明](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v4-to-v5)：SDK v5 要 Python 3.11，现适配器固定 4.8.1；自托管 v3 最低 3.63.0，v4 的读回 API 仍须按实际服务版本核验。机器上 Docker CLI 存在，但检查时 Docker Desktop Linux Engine 不可用，若干常见本机端口未监听；这不证明其他本机或远端实例不存在。

## 实现与可视化

- 一个 Turn 使用由 run/session/turn 确定的 32 位 Trace ID；无 run 使用交互命名空间。同一目标及共享标记目录下先占位、最多尝试一次建树；不同实验和评分 ID 隔离。同一 SQLite 会话的 turn 幂等是另一层约束，不同实验须使用独立新记忆目录。
- Turn → Stage → LLM request 按 invocation_id 连接。重复阶段以次数区分，状态映射到 level，并保存安全元数据中的业务起止与耗时。SDK 原生 span 时间是异步投影时间，不是业务瀑布时间。
- 只投影计数、白名单状态、候选及证据 ID 哈希、子目标候选覆盖、记忆召回/压缩/修复和白名单降级操作；覆盖不表示语义支持。Provider 有完整 usage 时按非重叠桶记录；没有 usage 为 unknown，不按字符估算。
- `eval e2e` 保留数据集哈希、配置摘要及哈希、实验和案例标识、评分来源；`execution_success`、引用结构或决策一致等规则结果可补到对应 Trace。同模型 Judge 明确区分；unknown 不输出为零。某评分先已知后变 unknown 时不会撤回远端旧值，本地标记 `unknown_not_retracted`。
- 外部遥测默认关闭。私有 OTel provider、仅本适配器登记的 span 白名单及 SDK 资源复用检查，防止意外发送其他 instrumentation；外部预占同公钥资源时停止导出。SDK 必需协议认证头和固定环境/版本元数据除外，问题、偏好、论文正文、密钥及任意错误响应正文不作为业务元数据上传。标识的无盐 SHA-256 是伪匿名，可被低熵枚举，不是匿名化。
- SDK 4.8.1 的分数采样可能受全局 OTel sampler 影响。队列容量1，业务线程等待默认0.5秒、最大2秒；进程退出、失败或超时可能丢失。初始化前占位和本地 flush 不是接收回执，不是全局精确一次；标记已存在时中断产生的不完整树不会自动重发，状态 `tree_already_claimed_unverified`。SDK 的内部实现仅针对固定版本验证。

## 回归与离线 SDK 检查

| 环境/命令 | 结果 |
|---|---|
| 正常环境：`.venv-gpu\Scripts\python.exe -m pytest -q --basetemp=.cache/pytest-langfuse-normal --junitxml=reports/revision/langfuse-full-normal.xml` | **264 passed、73 warnings，25.72秒**；FAISS 和本机 HTTP 测试可用。[日志](langfuse-full-normal.log)、[JUnit](langfuse-full-normal.xml)。 |
| 受限环境：同命令使用 `pytest-langfuse-restricted` 和对应 JUnit | 测试摘要 **256 passed、8 failed、73 warnings，55.04秒**；1项 FAISS DLL 初始化失败、7项本机 HTTP 权限失败；进程随后异常退出 `-1073741819`，日志还记录 Windows fatal exception `0xc0000008`，不推断具体根因。[日志](langfuse-full-restricted.log)、[JUnit](langfuse-full-restricted.xml)。不能直接归因为业务缺陷。 |
| SDK 路径启用的定向 pytest：`tests/test_revision_observability.py`、`test_langfuse_delivery_status.py`、`test_followup_trace_paths.py`、`test_telemetry_acceptance.py`、`test_eval_telemetry_provenance.py` | **37 passed，5.32秒**；包括实际 SDK 的内存 exporter 短进程和卡住 flush 测试。[日志](langfuse-targeted-sdk.log)、[JUnit](langfuse-targeted-sdk.xml)。不代表任意环境下所有退出情形。 |
| `scripts/check_langfuse_sdk_offline.py`，PYTHONPATH 指向隔离 SDK 和 src | **离线通过**。实 SDK、内存 exporter、网络禁止；2 次不同 run、同 run 重放、过滤非本适配器 Span、分数序列化及映射。Score queue 截获，未请求服务端。 |

此前 SDK 探针 [v2 日志](langfuse-sdk-offline-v2.log) 曾因检查 `ScoreBody.trace_id` Python 属性失败；检查 SDK 实际发送前的 `model_dump(exclude_none=True)` 后，确认序列化中含 `traceId`，后续 [v4](langfuse-sdk-offline-v4.log) 与 final 探针通过。服务端是否接受及归属仍需读回，不把离线序列化等同服务器验证。此前 [preliminary 定向日志](langfuse-targeted-preliminary.log) 保留。

## 固定合成样本与下一步

离线 [合成数据](langfuse-synthetic-fixture/manifest.json) 为 1 个案例、1 篇合成论文及 BM25 索引：数据哈希 `da7d022df22967fc8488bfceb9e74e6bcee18e9b7188423ea74b424180da8c87`，语料哈希 `ca89c39d61145561dbebab7de637f654841fb9cdad550eededb811d3209c841e`。未运行真实问答；规则不是人工语义质量标签。准备器不会覆盖已有目录或发出请求。真实验收建议固定该样本运行两个新目录，每组最多3次模型请求、合计最多6次、每次最多1024输出 token、45秒、无重试或修复；记录所有实际请求和 provider usage，再按本地 Trace 映射读回层级、状态、usage、检索及记忆摘要、规则 Score，并比较不同 run 与重放。具体命令和带截止时间的官方读回接口见 [追踪文档](../../docs/runtime-tracing.md)。

最少缺失条件：**可访问的 Langfuse Cloud 或兼容自托管实例的地址、有效公钥与密钥，以及上传该合成测试元数据的明确授权**；还须有已配置且授权调用的小预算模型。凭据应由用户放入本机安全环境变量，不需要在聊天中提供。服务端 v4 建议按官方 [SDK 查询文档](https://langfuse.com/docs/api-and-data-platform/features/query-via-sdk) 读取 observations 和 scores；实际版本、配置摘要、请求数、token、失败、远端内容与本地对应都必须另行保存。仅完整服务端读回成功才可称真实端到端验收通过。

本轮未扩大三类记忆、拆分或熔断行为；拆分和熔断仍默认关闭，没有已验证的独立备用模型。旧 `runs/` 46对会话文件归属未确认且未导入，原 92 份备份保留，见 [此前核验](closure-backup-check.json)；本地命名空间不是身份认证。
