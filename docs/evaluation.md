# 评测协议与复现

当前开发集固定在 `data/eval/v0.3/`。旧 `data/eval/*.jsonl`、v0.2 和旧运行产物保留；不能混合不同版本结果。QASPER 官方检索标注可用于段落覆盖评估，其余标签仍为 `draft_unreviewed`。静态校验清单在 `reports/acceptance/data-audit.json`，不替代人工复核。

## 历史与分母

`prepare_eval.session_history` 仅按同一 session 的严格较早 turn 构造历史，排序后保存 role/content/turn_id；单轮历史为空。历史是用户问题上下文，不伪造助手答案。expected_query 缺失保持 null，不编造 standalone gold。Rewrite token overlap 只是诊断，语义保持需辅助 Judge 或人工评估。

Boundary 默认 `--boundary-mode replay`：依次执行历史用户问题，使用真实生成的助手答案再执行目标问题。`context-only` 则注入指定消息。每条案例使用独立执行会话，统计聚类仍按原 session。两个模式不可混合汇总。Replay 历史失败导致目标失败，仍保留该案例在分母中。

## 成功判定

`eval_protocol.assess` 分开记录 execution_success、task_success、pipeline_success：

- execution：未发生未处理异常且 trace completed。
- task：最终 decision 与预定义条件一致；应回答案例还要求独立 reviewed_answer_correct。没有质量标签时为 null，不以引用存在代替正确性。
- pipeline：execution、输出 schema、应执行阶段、引用 ID/来源结构校验（回答时）或合适拒答、Memory 提交与 task 全部通过。

所有原始案例都在分母中，未知质量不计为成功，并单列 unknown。此时成功率是已确认成功的下界，不是完整质量估计。behavior_match 单独报告草稿行为标签一致率。另记录 NLU、Reflect（未有独立标签时 null）、检索有效性、生成和记忆提交。degraded 不自动代表任务正确。模型 Judge 仅产生 provisional 指标，不填入人工 reviewed 字段。

## 配对统计

`bootstrap_ci` 固定 seed=42、默认 2000 次；以 session 为多轮抽样单位，检索以 source paper 为单位。有放回抽取 cluster，重复抽中的所有行重复计权。均值为 turn 加权均值。小簇数量（尤其单会话）区间不能作总体推断。

`eval compare RUN_A RUN_B --metric behavior_match --output comparison.json` 读取逐例文件，以 session/case ID 配对，列共同样本、单侧缺失、错误、未知指标、改善与退化，以及 B-A 的簇配对 bootstrap 95% CI。失败按零计，未知质量单列，配置缺失/不一致明确标记。

## 检索口径

四组固定相同 corpus/index/query hash、30 候选。Recall@5/10 为 gold 段落覆盖宏平均，多替代标注取最佳路径；任一窗口命中为覆盖代理。MRR@5/10 为首个相关 chunk 倒数排名。nDCG@5/10 使用 chunk 二元相关性（与任一 gold 段落交集），IDCG 使用完整语料相关 chunk 数；窗口重复相关是该代理的局限。失败计零。检索实验不调用 API。

## 实际可执行消融

`eval ablation --config configs/evaluation.json --output NEW_DIRECTORY` 读取 ablations[].args，依次调用相应评测，逐实验保存配置、指标和结果。单组失败会记录并继续。无执行开关的 no-citation-validation 已移除。配置文件的其他设计字段不是所有 CLI 参数的隐式覆盖器，实际执行以 args 与 run_config.json 为准。

Reflect 固定前 3 个 session（12 turns）、BM25，比较不反思、一次反思、允许二次检索；一次反思与禁止二次检索是同配置重复对照。Memory 固定前 2 组十轮会话，比较完整历史、确定性裁剪、语义压缩、语义压缩加召回。不是人工金标效果实验。

## 原工作目录复现

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'
& $py -m pytest -q
& $py scripts/audit_eval_data.py
& $py scripts/retest_budget.py --output reports/new-budget
& $py scripts/run_retrieval_suite.py --reuse-index --output reports/new-retrieval
& $py scripts/report_results.py --run reports/new-retrieval
& $py -m paper_agent.cli eval intent data/eval/v0.3/intent.jsonl --output reports/new-intent --max-tokens 4096 --thinking disabled
& $py -m paper_agent.cli eval rewrite data/eval/v0.3/query_rewrite.jsonl --output reports/new-rewrite --max-tokens 4096 --thinking disabled
& $py -m paper_agent.cli eval boundary data/eval/v0.3/boundary.jsonl --index indexes/bge --mode bm25 --boundary-mode replay --workers 4 --max-tokens 4096 --thinking disabled --output reports/new-boundary
& $py -m paper_agent.cli eval e2e data/eval/v0.3/dialogs.jsonl --index indexes/bge --mode rerank --workers 4 --max-tokens 4096 --thinking disabled --output reports/new-e2e
& $py -m paper_agent.cli eval ablation --output reports/new-ablations
& $py scripts/judge_acceptance.py reports/new-e2e data/eval/v0.3/dialogs.jsonl
```

直接复用 data、models、indexes、.env，不需要重新下载、重建或覆盖凭证。70-turn 是完整对话运行后的计分视图，不能省略所依赖的历史轮次。真实 token 包含失败和重试；无 usage 的超时请求成本未知，不能报告成零。P50/P95 是本机当次延迟，不是服务保证。所有实际结果、限制和未完成项见 reports 中的本轮报告。

本轮实际 API 数据冻结为 v0.3；后续 v0.4 补充显式 history session/turn 元数据、论文 ID/标题别名和实体指标适用范围。v0.4 的 Rewrite 分数是对保存的 v0.3 预测离线重算（`reports/acceptance/rewrite/rescored-v04.json`），不是新 API 实验；两版均保留草稿标签与原始结果。
