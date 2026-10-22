# 记忆压缩与召回评测

三层记忆保留。v0.3 有 8×10 轮长会话；真实消融固定前 2 个 session（20 turns/组），其余未执行，不能声称 80 个不同会话问题均已实测。检查点是讨论论文、事实证据来源，非人工事实文本金标。

| 组 | 检查点通过/总数 | 压缩轮 | 召回轮/记录 | API次数 | 已知token | P50/P95 ms | Judge质量/95% CI |
|---|---|---|---|---|---|---|---|
| ablations/memory-bounded | {'correct': 20, 'total': 20, 'rate': 1.0} | 12 | 0/0 | 62 | 394238 | 13869/18136 | 95.00% [90.00%, 100.00%] |
| ablations/memory-full | {'correct': 20, 'total': 20, 'rate': 1.0} | 0 | 0/0 | 61 | 430178 | 14499/18387 | 100.00% [100.00%, 100.00%] |
| ablations/memory-recall | {'correct': 20, 'total': 20, 'rate': 1.0} | 12 | 10/37 | 71 | 554433 | 15047/21364 | 未知 |
| ablations/memory-semantic | {'correct': 20, 'total': 20, 'rate': 1.0} | 12 | 0/0 | 73 | 525947 | 17177/21700 | 95.00% [90.00%, 100.00%] |
| memory-recall-v2 | {'correct': 20, 'total': 20, 'rate': 1.0} | 12 | 10/54 | 72 | 556321 | 14858/19614 | 85.00% [80.00%, 90.00%] |

memory-recall-v2 是修复两次召回覆盖问题后的独立重跑，旧 memory-recall 保留；应使用 v2 与 semantic 对照。压缩 tokens 是实际全阶段 usage（包括摘要/重试），不能只用字符估计声称成本节省。token 比较和 checkpoint 配对 CI 见 acceptance/summary.json、paired-comparisons.json。

已验证的机制：长会话早期消息移出近期窗口后可以召回；同 session 隔离；同 key 新事实优先且旧记录标记 superseded；重复 turn 与进程恢复不重复提交；归档写失败不部分提交。相关证据来自 test_reliability.py，属于 Mock/离线功能测试。

尚未独立验证：真实模型对人工标注事实更新/冲突、归档召回精确率、早期实体指代准确率、错误事实率的完整金标质量。此次长集重复问题较多，工作 facts 可能未超 40，不能用其 checkpoint 满分证明任意长对话记忆可靠。
