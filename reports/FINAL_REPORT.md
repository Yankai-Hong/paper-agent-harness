# nano-cc 本轮最终交付报告

结论：工程正确性修正与核心开发实验已完成；人工金标、独立语义复核和充分的长期记忆质量验证未完成，尚不满足全部科研验收条件。旧报告保存在 history/pre-acceptance，所有结果均来自本目录实际产物。

## 实际修改

| 文件/函数 | 修改 |
|---|---|
| scripts/prepare_eval.py · session_history/prepare | 同 session 严格过去历史；版本保护；长记忆非空检查点；ID/标题别名与适用范围 |
| eval_tasks.py · bootstrap_ci/run_boundary/run_rewrite/run_memory | 正确有放回簇抽样；Replay/Context-only；拒绝污染历史；逐轮记忆检查 |
| eval_protocol.py · assess/success_metrics/compare_runs | 三类成功率、未知质量、逐例配对差值与 CI、缺失/失败/配置报告 |
| eval_cli.py · cmd_ablation/cmd_compare/cmd_e2e | 真实执行矩阵、保存有效配置、会话并行与逐会话结果、API参数覆盖 |
| providers.py · APIConfig/post_json/CompatibleReasoner | 连接与总超时、有限重试/结构修复、状态分类、token/未知成本记录 |
| pipeline.py · PaperAgentState/PaperPipeline.run/compact_state | 单快照原子提交、turn_id 幂等/回滚、组件降级、归档上下文与合并修复 |
| memory.py · SessionMemory.load/ContextArchive.search | 新旧快照读取、会话隔离、关键词/实体/论文/时间召回、旧事实标记 |
| compaction.py · semantic_compact_state | 保留讨论论文元数据、归档 turn 序、压缩观测字段 |
| evaluation.py · retrieval_metrics/evaluate_retriever | Recall@10、MRR、nDCG 及完整排名 |
| tests/test_eval_corrections.py、test_reliability.py、test_providers.py | 统计、历史、幂等、降级、超时、归档和长对话回归 |
| 新增 retest_budget/audit_eval_data/judge_acceptance/review_rewrite/quality_statistics/summarize_acceptance/finalize_acceptance 脚本 | 真实探针、静态审核、辅助评审、统计与可重复生成报告 |
| configs/evaluation.json、README、docs 全部组件页 | 执行矩阵、协议、运行边界和实测配置同步 |

七阶段及原 Harness、BM25/BGE/FAISS/RRF/Cross-Encoder 保留；未下载模型、重建原索引或替换 .env，未新增外部数据库/服务。

## 验证与数据

原依赖环境最新 61 项测试全部通过，见 acceptance-tests.xml。Mock 与真实调用分开。真实数据为 v0.3（Intent 100、Rewrite 90、Boundary 55、30×4 E2E、Memory 子集 2×10）。v0.4 仅改进历史元数据/实体别名标签，保留草稿状态；其 Rewrite 使用 v0.3 保存预测做离线重新评分，未冒充新 API 实验。

OpenCode t3：2048 截断，4096/8192 成功；4096+disabled 实测 reasoning_tokens=0。四轮回归4/4完成；全量 execution 120/120，70-turn 视图从完整历史运行取分。全量阶段 API 298 次、已知 token 1574990，P50/P95 12386/22473 ms。

E2E 行为与草稿标签一致率 74.17% [66.67%, 80.83%]。严格规则 task/pipeline 已满足条件下界均 30/120，另 59 条质量未知；不能把 completed/total 当综合成功率。辅助 Judge 答案质量 85.00% [78.33%, 90.83%]；结合草稿行为条件的临时 Pipeline Success 70.00% [61.67%, 77.50%]。这些不是人工验证的正式指标。

引用 ID 与来源结构均 499/499；语义支持 437/439，Unsupported Claim Rate 0.46%（同模型评审）。全部执行/质量失败和不一致案例保存，而不是从分母移除。

## 算法结果与成本

检索 Recall@5：66.26% / 73.94% / 74.24% / 77.61%，新四组复测与旧记录一致。Hybrid−Dense +0.30pp，95% CI [-7.40,+6.95]；Rerank−Hybrid +3.38pp，CI [-4.29,+11.04]，均不能声称显著。完整 Recall@10/MRR/nDCG、互补案例与排名变动见 ABLATION_RESULTS.md。

Reflect 12 turns：禁用/一次反思/允许二检的辅助答案质量为 83.33% / 83.33% / 66.67%，未证实二检增益。Memory 20 turns/组：完整/裁剪/语义/语义+召回v2 的辅助质量为 100% / 95% / 95% / 85%；结构检查点均20/20，不能当作事实语义全正确。

| 比较 | 辅助质量差值/95% CI | 全阶段token变化 |
|---|---|---|
| ablations/reflect-retrieve-again minus ablations/reflect-disabled | -16.67% [-25.00%, 0.00%] | +35.64% |
| ablations/reflect-retrieve-again minus ablations/reflect-enabled | -16.67% [-25.00%, 0.00%] | +4.67% |
| ablations/memory-bounded minus ablations/memory-full | -5.00% [-10.00%, 0.00%] | -8.35% |
| ablations/memory-semantic minus ablations/memory-full | -5.00% [-10.00%, 0.00%] | +22.26% |
| memory-recall-v2 minus ablations/memory-semantic | -10.00% [-20.00%, 0.00%] | +5.78% |

成本账本：已保存请求下界 1681 次，已知 token 7988049，usage 未知 7 次。包括诊断、重跑和 Judge，不能当作一次部署对话的成本。剩余额度与币值成本未知，详见 acceptance/cost-ledger.json。

## 复现与剩余验收

完整命令见 docs/evaluation.md。原目录使用 `.venv-gpu/Scripts/python.exe`；已有 data/models/indexes/.env 直接复用。先 pytest，再 smoke/小回归；检索用 --reuse-index；真实任务显式 --max-tokens 4096 --thinking disabled。报告重算不调用 API：

```powershell
.venv-gpu/Scripts/python.exe scripts/quality_statistics.py
.venv-gpu/Scripts/python.exe scripts/summarize_acceptance.py
.venv-gpu/Scripts/python.exe scripts/finalize_acceptance.py
.venv-gpu/Scripts/python.exe scripts/check_artifacts.py
```

待完成项目见 REMAINING_ISSUES.md：人工金标/引用复核、正式证据不足子集、真实长期事实更新与召回精确率、更多自然长会话。已执行核心消融不等于已证明增益；不把未审核草稿、Mock 或参考截图当项目科研成果。
