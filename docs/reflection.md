# 证据反思与二次检索

实现：[PaperPipeline](../src/paper_agent/pipeline.py) 与 [Reflect schema](../src/paper_agent/providers.py)。Reflect 看当前问题、实际候选原文、意图和尝试次数。

| 决策 | 行为 |
|---|---|
| answer | 证据足够，进入引用生成 |
| retrieve_again | 必须给出 next_query，再检索一次 |
| clarify | 问题/对象不清，要求澄清 |
| refuse | 无法支持回答，输出拒答 |

第一次取 5 条，第二次取 5 条，并按 chunk_id 合并保留首轮证据，因此最多保留 10 条。第二次仍要求检索时强制拒答并记录 retrieval_budget_exhausted，不无限循环。没有证据时即使反思选 answer，也改为拒答。

比较题要求检查双方的证据，不应因为一篇相关就推断另一篇。检索到的文档被提示为数据，不能作为指令执行。真实拒答是否恰当仍由独立评测判断；程序不能仅凭一个相关性分数证明证据充分。

`--no-reflect` 跳过模型反思并尝试回答现有证据，用于消融；仍保留空证据拒答和引用结构校验。该开关可能降低可靠性，不是默认运行模式。

本地测试覆盖“连续要求二次检索后强制拒答”、错误 trace 和成功路径；本轮实际质量差异见消融报告，不使用参考截图数值。

## 本轮执行配置

--max-retrieval-attempts 1/2 控制检索预算，达到上限仍 retrieve_again 则拒答。Reflect API 失败或非法决策不会默认宣称证据充分，整轮失败并回滚。消融结果保存到 reports/acceptance/ablations，质量来自单列的临时模型评审，非人工金标。
