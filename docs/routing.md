# 路由、指代消解与 Query 改写

实现：[providers.py](../src/paper_agent/providers.py) 的 NLU schema/提示词，以及 [pipeline.py](../src/paper_agent/pipeline.py) 的控制流。这部分是新增领域实现，不是 learn-claude-code 内置功能。

NLU 输入当前问题、近期消息、结构化会话记忆及论文 ID/标题目录。输出字段：

```json
{"intent":"fact","standalone_query":"standalone question in corpus language","needs_clarification":false,"paper_ids":["actual-paper-id"],"resolved_entities":{"它":"actual-paper-id"}}
```

意图有 search（找证据）、fact（事实问题）、explain（解释）、compare（比较）、out_of_scope（库外请求）五类。前四类共用检索流程，并非五套独立 Agent。越界直接拒答；指代歧义需要澄清；其余进入检索。论文 ID 必须在目录中，未知 ID 报错，不在错误范围内悄悄检索。

改写目标是把“它”“刚才那篇”等补成独立问题，并尽量采用语料语言。中文问题面对英文语料时，依赖模型进行适当改写。`--no-rewrite` 保留 NLU 分类和范围识别，仅让检索直接使用用户原句，便于单因素消融。

目前 `resolved_entities` 是模型输出的字符串映射；ID 范围对 paper_ids 做强校验，实体语义仍需人工/真实模型验证。目录全部传入 NLU 适合当前 100 篇规模；更大规模需先做论文目录检索。输入超过预算会明确报错。

测试验证字段类型、未知意图、未知 ID、流程分支和会话恢复；没有用脚本响应算出 NLU 准确率。多轮草稿中的 provisional_intent 不是人工 gold。

## 历史召回与失败

NLU 输入额外包含 historical_context_not_evidence：有 session 隔离、条数/字符上限及 superseded 标志的归档记录。一次 NLU 后可根据 standalone_query 再查归档，但不循环 NLU。结构失败最多一次修复，仍无效则报错，不生成未经校验的路由。Rewrite 数据历史仅来自同 session 更早 turn，旧污染版本仅保留作历史记录。
