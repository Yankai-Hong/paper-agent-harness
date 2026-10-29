# 追踪身份、论文范围和有限评测：跟进验收

后续范围修复、最终源码真实集成与多轮对照见 [验收收尾](CLOSURE.md)；本文件保留本轮原始口径。

本轮在已有未提交工作区上继续，保留历史报告、模型运行原始记录与迁移备份，没有重置或推送。先前结论与数据保留在 [FINAL.md](FINAL.md)，本文件记录其后的修复和验证。

## 正确性修复

| 问题 | 原行为 | 当前行为与验证 |
|---|---|---|
| 不同实验同 session/turn 的 Trace/Score 碰撞 | 远端 Trace ID 仅由 session_id + turn_id 派生；Score 也可能共用。 | 以版本化、无拼接歧义的编码包含 run_id、session_id、turn_id；Trace ID 为32位非零十六进制，Score ID 包含 Trace ID、来源、名称。同一运行重放稳定，跨运行隔离。普通交互无 run_id 时使用独立 interactive 命名空间，按 session/turn 稳定映射。 |
| 本地同目录不同运行覆盖 | 同一 turn_id 均写同名文件。 | 有 run_id 写 `runs/<run哈希>/<turn>.json`，交互无 run_id 保持旧路径；Pipeline 返回 trace_path，评测和补分读取该路径。Trace 存 `observability_mapping`，旧 Trace 缺失或映射不符时补分仅留本地并标记 withheld，不猜测远端归属。 |
| 普通提及加历史指代误澄清 | 本轮显式论文集合必须等于 NLU 范围，合法补充历史论文也被拦截。 | 普通提及必须包含明确论文；出现历史指代时可增加本会话实体、讨论摘要、历史消息或引用可核对的论文。未核实扩展或漏掉显式对象则澄清。仅限范围须严格一致；含历史指代的仅限和排除条件暂保守澄清。开关关闭与开启均适用，最终引用只来自本轮证据。 |

规则仍无法完整理解所有自然语言范围；复杂排除和排他范围可能要求额外澄清，不能宣称通用指代正确。测试包括同 case 跨运行与 Score 隔离、幂等重放、补分、无映射旧 Trace、同目录不覆盖、多轮指代、仅限、排除、越权、两种拆分状态与引用边界。

## Langfuse 与隐私

本地 Trace 仍为权威，默认关闭外部遥测。白名单新增候选/重排/证据数量、证据 ID 哈希摘要、子目标覆盖计数、记忆召回和修复数量以及已知降级原因；不发送问题、完整论文、偏好、密钥或错误正文。覆盖表示候选存在，不代表语义支持。规则分数和同模型 Judge 分数按来源关联；unknown 不换算为零。

本机未发现配置完整的 Langfuse 实例及凭据，也没有启用遥测；**服务端读回未验收**。SDK 替身与故障隔离回归通过，不能据此宣称真实服务验收。格式与确定性 ID 参考 [Langfuse 官方 Trace ID 文档](https://langfuse.com/docs/observability/features/trace-ids-and-distributed-tracing)。

SDK 不支持指定 observation ID：仅稳定 Trace ID 不足以防止重放时新增重复子树。现以目标实例/项目和 Trace ID 为键，在共享目录原子占位后最多尝试一次建树。不同机器或标记目录不共享去重；发送期间崩溃或结果不明时不会自动重发，远端可能缺失或只有部分树，标记不证明服务端成功。评分仍以稳定 Score ID 独立更新。此为保守的本地最多一次尝试，不是经实测的全局精确一次，详见 [运行追踪](../../docs/runtime-tracing.md)及 [SDK 官方文档](https://langfuse.com/docs/observability/sdk/instrumentation)。

## 测试环境

正常环境可访问本机 HTTP 测试服务并加载 FAISS。遥测幂等补强后的最终完整回归为 **181 passed，73 warnings，19.07 秒**；[日志](followup-final-tests.log)、[JUnit](followup-final-tests.xml)。此前同轮修复的完整回归为179通过，保留于 [followup-tests.log](followup-tests.log)。用户提及此前受限环境为143通过、8失败，但该次原始日志未在当前 reports 中找到，作为用户提供的既往记录保留，不冒称自行复现同一运行。

本轮独立受限环境重跑：**171 passed，8 failed，73 warnings，48.41 秒**；其中1项 FAISS DLL 初始化受限、7项本机 HTTP 权限失败，见 [日志](followup-restricted-tests.log) 和 [JUnit](followup-restricted-tests.xml)。该受限测试发生在最后增加两个遥测幂等测试之前；当时相同179项在正常环境全部通过。两种环境结果分别报告，不把受限环境问题直接归因为业务逻辑缺陷。`git diff --check` 和历史报告/数据校验另行通过。

## 模型对照：公开样本，固定预算

预先固定 [协议](followup-experiment-protocol.md)：QASPER v0.6 前两个不同论文，构造简单、双论文和缺失证据三种问题，各自拆分关闭/开启；真实本地 BM25，`deepseek-v4.1-flash`，thinking disabled，最多18个实际请求、每轮3个、输出4096 token、无传输重试、Schema 修复或 Judge，最多一轮检索。没有使用个人记忆或 Langfuse 遥测。理论输出上限73,728 token；费用金额未知。实际 **14 次请求**，所有6轮均完成并提交，provider usage 14/14 可用；[逐轮结果](model-comparison-followup/summary.json)、[源码和数据哈希](model-comparison-followup/run-freeze.json)。

该模型实验结束后只对 `observability.py` 增加了上文的建树去重；实验始终关闭遥测，没有重新运行付费样本。源码哈希保留实验当时的版本，最终完整测试覆盖后续修复。

| 固定样本 | 拆分关闭 | 拆分开启 | 证据目标检查 |
|---|---|---|---|
| 单论文简单题 | answer；3次、15,645 token、14.06秒 | answer；3次、15,051 token、13.47秒 | 两组目标论文各1/1，官方证据段落各1/1；检索均1次 |
| 双论文组合题 | refuse；2次、8,906 token、7.77秒 | refuse；2次、8,513 token、7.22秒 | 两组目标论文各2/2，官方证据段落各1/2；检索1→2次 |
| 缺失证据探针 | refuse；2次、8,664 token、6.77秒 | refuse；2次、8,601 token、7.03秒 | 两组目标论文各1/1；目标训练数据校验和的语义标注为 unknown |

汇总各3轮：关闭/开启均7次请求，已知 token 33,215/32,165，耗时28.61/27.72秒。token 汇总是 provider 返回的非缓存输入、缓存命中和输出之和，不重复计缓存；时延和成本差异只描述本次顺序执行，金额未知。官方证据段落命中两组均2/3，缺失探针不在该分母内。对照样本太小且没有独立语义标签，答案完整性、无依据结论和误拒答均为 unknown；引用 ID 合法不等于语义支持，不能声明质量提升。

此前 [model-pair/summary.json](model-pair/summary.json) 中 `budget.actual_requests=6` 实际是预算上限，逐组实际2+3=5次；原文件哈希与本轮前一致，未覆盖。它只有一组完成，输出上限1024且 thinking 未禁用，与本轮不能作为同配置直接对比。后续脚本将 `max_actual_requests` 和 `actual_requests_total` 分开记录。

## 旧数据和剩余事项

复核发现的46对旧 history/记忆文件仍全部未确认归属；92个备份文件均存在且 SHA-256 匹配，46个迁移演练目标中导入会话数为0。[基线核对](followup-baseline.json)、[备份清单](file-migration-inventory.json)。此结论不是个人数据迁移完成；如要恢复这些旧会话，需先由用户确认具体来源及归属，再按 [迁移文档](../../docs/memory.md) 在新空目标操作。已确认的历史评测库迁移与个人旧文件分开。

本地 user/project/session 命名空间按存储根目录隔离；项目跨会话共享是显式选项，不提供身份认证或抵御具有本机文件访问权限的其他操作者。独立备用 provider/model 的能力与切换未实现、未验证；自然语言偏好提取、跨会话指代和长期记忆质量尚无独立人工评测。

**判断：本轮两项修复和本地工程回归达到验收条件，有限模型对照已执行并保留未知项；真实 Langfuse 服务端读回、备用模型及自然语言记忆质量仍未验收，不能称为全面端到端或生产高可用验收。**
