# 路由、指代消解与 Query 改写

实现：[providers.py](../src/paper_agent/providers.py) 的 NLU schema/提示词，以及 [pipeline.py](../src/paper_agent/pipeline.py) 的控制流。这部分是新增领域实现，不是 learn-claude-code 内置功能。

NLU 输入当前问题、近期消息、结构化会话记忆及论文 ID/标题目录。输出字段：

```json
{"intent":"fact","standalone_query":"standalone question in corpus language","needs_clarification":false,"paper_ids":["actual-paper-id"],"resolved_entities":{"它":"actual-paper-id"}}
```

意图有 search（找证据）、fact（事实问题）、explain（解释）、compare（比较）、out_of_scope（库外请求）五类。前四类共用检索流程，并非五套独立 Agent。越界直接拒答；指代歧义需要澄清；其余进入检索。论文 ID 必须在目录中，未知 ID 报错，不在错误范围内悄悄检索。

普通显式论文提及要求 NLU 结果包含该论文。历史指代仅从当前会话可核对的用户消息和助手引用判断：无锚定“它”只在候选唯一且历史未被压缩遮蔽时使用；“刚才/上一轮”依据最近一轮，无法确定则澄清；“前者/后者”需最近一轮有明确且一致的两篇论文顺序。检索候选汇总的摘要和模型推断实体不能单独证明指代。NLU 论文范围、可识别的改写对象与指代实体必须一致，否则澄清。

`仅限` 论文必须与明确对象一致；排除论文或无法判断的排除条件保守澄清。可识别的时间/数据排除与仅用某年份数据作为原始约束保留在检索、反思和生成的输入中；没有 Reflect 时澄清，有 Reflect 也不能把其判断当作经过独立核验的语义证明。规则识别精确 ID 和完整标题，复杂自然语言仍可能误判或额外澄清。无论拆分开关状态，最终引用必须来自本轮检索证据。

改写目标是把“它”“刚才那篇”等补成独立问题，并尽量采用语料语言。中文问题面对英文语料时，依赖模型进行适当改写。`--no-rewrite` 保留 NLU 分类和范围识别，仅让检索直接使用用户原句，便于单因素消融。

目前 `resolved_entities` 是模型输出的字符串映射；ID 范围对 paper_ids 做强校验，实体语义仍需人工/真实模型验证。目录全部传入 NLU 适合当前 100 篇规模；更大规模需先做论文目录检索。输入超过预算会明确报错。

测试验证字段类型、未知意图、未知 ID、流程分支和会话恢复；没有用脚本响应算出 NLU 准确率。多轮草稿中的 provisional_intent 不是人工 gold。

## 历史召回与失败

NLU 输入额外包含 historical_context_not_evidence：有 session 隔离、条数/字符上限及 superseded 标志的归档记录。一次 NLU 后可根据 standalone_query 再查归档，但不循环 NLU。结构失败最多一次修复，仍无效则报错，不生成未经校验的路由。Rewrite 数据历史仅来自同 session 更早 turn，旧污染版本仅保留作历史记录。
