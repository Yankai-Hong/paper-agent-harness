# 本轮真实 OpenCode 评测

原 provider/.env 未替换；真实调用与离线故障注入分开。旧报告保留于 history/pre-acceptance。开发批次使用 deepseek-v4.1-flash、4096、thinking=disabled；所有草稿/模型 Judge 指标均 provisional。

t3 固定输入预算对照见 acceptance/budget-v2/results.json：2048 截断；4096/8192 成功；disabled 实测 reasoning_tokens=0。此前连接超时实现缺陷造成的失败保留在 budget/；中断中的请求成本可能未知。四轮端到端 smoke 完成 4/4。

Intent 100 条：Accuracy 85.00% [78.00%, 91.00%]，Macro-F1 0.8267；完整混淆矩阵和 15 个不一致案例见 intent/。
Rewrite 90 条：原 ID 字面匹配指标 0.00% 不等同于实体丢失。新增别名诊断全部 90 条 75.56% [70.56%, 80.56%]；实体适用的 t2/t3 共 60 条 98.33% [95.83%, 100.00%]。保留原始结果及口径变更，不改样本或补造 expected_query。语义/指代与检索增益见 rewrite/semantic-summary.json，均为辅助或来源证据代理。
Boundary Replay 55 条：一致率 90.91% [80.70%, 98.25%]；拒答 Precision 80.00%、Recall 100.00%、F1 0.8889；5 个不一致案例 ['d05-t4', 'd04-t1', 'd05-t1', 'd08-t1', 'd10-t1']。Context-only 没有混入本表。

## 运行与用量（阶段调用含重试，不含单列辅助 Judge）

| 运行 | 执行轮/调用 | 已知总 token | 未知 usage 调用 | P50/P95 ms | 降级轮 |
|---|---|---|---|---|---|
| ablations\memory-bounded | 20/62 | 394238 | 0 | 13869/18136 | 0 |
| ablations\memory-full | 20/61 | 430178 | 0 | 14499/18387 | 0 |
| ablations\memory-recall | 20/71 | 554433 | 0 | 15047/21364 | 0 |
| ablations\memory-semantic | 20/73 | 525947 | 1 | 17177/21700 | 0 |
| ablations\reflect-disabled | 12/21 | 108294 | 0 | 9621/13798 | 0 |
| ablations\reflect-enabled | 12/28 | 140337 | 0 | 14419/22509 | 0 |
| ablations\reflect-no-second | 12/27 | 130739 | 0 | 9819/13326 | 0 |
| ablations\reflect-retrieve-again | 12/30 | 146895 | 0 | 10494/17553 | 0 |
| boundary-replay | 145/380 | 1890572 | 0 | 11677/19484 | 0 |
| e2e-full | 120/298 | 1574990 | 0 | 12386/22473 | 0 |
| e2e-smoke | 4/10 | 54163 | 0 | 17013/27145 | 0 |
| intent | 0/100 | 51502 | 0 | 3572/10941 | 0 |
| memory-recall-v2 | 20/72 | 556321 | 0 | 14858/19614 | 0 |
| rewrite | 0/91 | 51696 | 0 | 3346/7161 | 0 |

完整 120 turns 和从其计分的 70-turn 视图分别保存在 e2e-full/ 与 e2e-70-metrics.json。execution/task/pipeline 含未知质量的下界不可混用。Citation ID Validity 与 Source Consistency 是结构检查；逐 claim 支持和 Unsupported Claim Rate 见 semantic-summary.json，是同模型评审且待人工抽查。未评审/失败案例不会被悄悄算作正确。

e2e-full：execution {'correct': 120, 'total': 120, 'rate': 1.0, 'unknown': 0, 'note': 'Unknown quality is not counted as success; denominator includes every case.', 'ci': {'mean': 1.0, 'n': 120, 'clusters': 30, 'bootstrap_95_percentile_interval': [1.0, 1.0]}}；task {'correct': 30, 'total': 120, 'rate': 0.25, 'unknown': 59, 'note': 'Unknown quality is not counted as success; denominator includes every case.', 'ci': {'mean': 0.25, 'n': 120, 'clusters': 30, 'bootstrap_95_percentile_interval': [0.25, 0.25]}}；pipeline {'correct': 30, 'total': 120, 'rate': 0.25, 'unknown': 59, 'note': 'Unknown quality is not counted as success; denominator includes every case.', 'ci': {'mean': 0.25, 'n': 120, 'clusters': 30, 'bootstrap_95_percentile_interval': [0.25, 0.25]}}。

e2e-smoke：execution {'correct': 4, 'total': 4, 'rate': 1.0, 'unknown': 0, 'note': 'Unknown quality is not counted as success; denominator includes every case.', 'ci': {'mean': 1.0, 'n': 4, 'clusters': 1, 'bootstrap_95_percentile_interval': [1.0, 1.0]}}；task {'correct': 1, 'total': 4, 'rate': 0.25, 'unknown': 3, 'note': 'Unknown quality is not counted as success; denominator includes every case.', 'ci': {'mean': 0.25, 'n': 4, 'clusters': 1, 'bootstrap_95_percentile_interval': [0.25, 0.25]}}；pipeline {'correct': 1, 'total': 4, 'rate': 0.25, 'unknown': 3, 'note': 'Unknown quality is not counted as success; denominator includes every case.', 'ci': {'mean': 0.25, 'n': 4, 'clusters': 1, 'bootstrap_95_percentile_interval': [0.25, 0.25]}}。
