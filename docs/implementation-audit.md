# 本轮源码验收审计

核对源码入口 PaperPipeline.run、APIConfig/CompatibleReasoner、ContextArchive.search、eval_tasks、eval_protocol、eval_cli 与 prepare_eval。旧审计保存在 reports/history/pre-acceptance/implementation-audit.md，仅用于历史比较。

| 项目 | 当前实现 | 证据/边界 |
|---|---|---|
| Rewrite 历史 | 同 session、严格较早 turn、排序消息 | test_eval_corrections；v0.3 数据 |
| Boundary | Replay 默认、Context-only 独立标记 | 历史实际调用；不同案例执行隔离 |
| Bootstrap | 重复簇按次数展开，固定种子 | 与独立抽样实现确定性对照 |
| 成功率 | execution/task/pipeline 分离 | 未确认语义为 unknown，不等于成功 |
| 消融 | 配置矩阵实际 dispatch、独立目录 | ablations/matrix_results.json |
| API | 建连/请求总超时、有限传输与结构重试 | 故障测试、t3 真实预算对照 |
| 降级 | Pipeline 重排失败保留检索排序、混合失败尝试单路 | 每轮 degraded/reason；独立检索基线保持严格失败 |
| Reflect 失败 | 失败并回滚，不默认充分 | 故障测试 |
| 引用错误 | 明确拒答、degraded | 结构与语义单独评估 |
| Memory | 稳定 ID、单快照原子提交、失败回滚 | 同 turn 重复/重启测试；非并发事务 |
| Archive | 同 session bounded recall，过期事实标记 | 接入 NLU/生成上下文；不当论文证据 |
| 数据 | 新版本、非空检查点、静态审阅清单 | 草稿不提升为人工 gold |

实际文件/函数改动、真实运行分母、token、延迟和未完成项以 reports/FINAL_REPORT.md 及配套报告为准。原七阶段名称、BM25/BGE/FAISS/RRF/Cross-Encoder、轻量 Harness 保留，没有引入额外服务或数据库。tool_loop 仍为独立 Harness，七阶段没有改成模型原生工具调用。
