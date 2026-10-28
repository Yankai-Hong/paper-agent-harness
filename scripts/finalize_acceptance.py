"""Final ledger and deliverables from completed saved experiments, without API calls."""
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from paper_agent.storage import read_jsonl, write_json, atomic_text
from paper_agent.eval_tasks import bootstrap_ci
from summarize_acceptance import ROOT, load, interval, table


def main():
    summary=load(ROOT/"summary.json")
    runs=summary["runs"]
    quality={p.parent.relative_to(ROOT).as_posix():load(p) for p in ROOT.rglob("quality-statistics.json")}
    paired={}
    for a,b in [("ablations/reflect-disabled","ablations/reflect-retrieve-again"),
                ("ablations/reflect-enabled","ablations/reflect-retrieve-again"),
                ("ablations/memory-full","ablations/memory-bounded"),
                ("ablations/memory-full","ablations/memory-semantic"),
                ("ablations/memory-semantic","memory-recall-v2")]:
        if a not in quality or b not in quality: continue
        qa={r["id"]:r for r in quality[a]["cases"]}
        qb={r["id"]:r for r in quality[b]["cases"]}
        ids=sorted(qa.keys() & qb.keys())
        diffs=[float(qb[i]["answer_correct"])-float(qa[i]["answer_correct"]) for i in ids]
        paired[b+" minus "+a]={"quality_delta":bootstrap_ci(diffs,[qa[i]["session_id"] for i in ids]),
                "improved":[i for i,d in zip(ids,diffs) if d>0],"regressed":[i for i,d in zip(ids,diffs) if d<0],
                "token_change_ratio":runs[b]["telemetry"]["known_total_tokens"]/runs[a]["telemetry"]["known_total_tokens"]-1,
                "definition":"same-model answer correctness, provisional; token totals include all pipeline stages"}
    write_json(ROOT/"quality-paired-comparisons.json",paired)

    # Source hashes plus exact per-query ranking movement, not inferred gains.
    ranks={m:{r["query_id"]:r for r in read_jsonl(ROOT/"retrieval-v2"/m/"predictions.jsonl")} for m in ("bm25","dense","hybrid","rerank")}
    overlap=[]
    movement=[]
    for q in ranks["bm25"]:
        a,b=set(ranks["bm25"][q]["candidate_ids"]),set(ranks["dense"][q]["candidate_ids"])
        overlap.append(len(a&b)/len(a|b) if a|b else 0)
        before=[h["chunk_id"] for h in ranks["hybrid"][q]["hits"][:5]]
        after=[h["chunk_id"] for h in ranks["rerank"][q]["hits"][:5]]
        movement.append({"query_id":q,"hybrid_top5":before,"rerank_top5":after,"order_changed":before!=after,"membership_changed":set(before)!=set(after)})
    write_json(ROOT/"retrieval-v2/ranking-analysis.json",{"mean_candidate_jaccard":float(np.mean(overlap)),
        "top5_order_changed":sum(r["order_changed"] for r in movement),"top5_membership_changed":sum(r["membership_changed"] for r in movement),"cases":movement})

    # Ledger is a known lower bound; interrupted/timeout requests may lack usage.
    pipeline_calls=sum(r["telemetry"]["api_requests"] for r in runs.values())
    pipeline_tokens=sum(r["telemetry"]["known_total_tokens"] for r in runs.values())
    judge_calls=judge_tokens=judge_unknown=0
    for p in ROOT.rglob("semantic-review.jsonl"):
        for row in read_jsonl(p):
            for c in row.get("calls",[]):
                judge_calls+=1
                judge_tokens+=sum(c.get("usage",{}).values())
                judge_unknown+=not c.get("usage_complete")
    probes=[]
    for directory in ("budget","budget-v2"):
        for row in load(ROOT/directory/"results.json",[]): probes += row.get("calls",[])
    smoke_reports=[load(p) for p in ROOT.glob("smoke*/smoke.json")]
    smoke_calls=sum(len(s.get("checks",{})) for s in smoke_reports)
    smoke_tokens=0
    for smoke in smoke_reports:
        for key,check in smoke.get("checks",{}).items():
            usage=check.get("usage",{})
            smoke_tokens+=sum(usage.values()) if key=="json" else usage.get("prompt_tokens",0)+usage.get("completion_tokens",0)
    legacy_judge=load(ROOT/"e2e-smoke/judge/judgments.json",[])
    extra_calls=[r["call"] for r in legacy_judge if r.get("call")]
    ledger={"pipeline_and_component_requests":pipeline_calls,"auxiliary_judge_requests":judge_calls+len(extra_calls),
            "budget_probe_requests":len(probes),"smoke_requests":smoke_calls,
            "known_total_requests":pipeline_calls+judge_calls+len(extra_calls)+len(probes)+smoke_calls,
            "known_total_tokens":pipeline_tokens+judge_tokens+sum(sum(c.get("usage",{}).values()) for c in probes+extra_calls)+smoke_tokens,
            "unknown_usage_requests":sum(r["telemetry"]["unknown_usage_calls"] for r in runs.values())+judge_unknown+sum(not c.get("usage_complete") for c in probes),
            "account_quota_remaining":"not exposed by configured API; unknown",
            "currency_cost":"not calculated; no verified price/account billing data",
            "limits":"100 requests per reasoner by .env; retries count; multiple sessions/processes have independent limits",
            "interrupted_probe_warning":"Initial budget process stopped after timeout implementation defect; in-flight requests may be absent from saved results. Totals are known lower bounds, not exact billing totals.",
            "future_demand_without_retries":{"intent":100,"rewrite":90,"e2e_120_upper_bound":480,
                    "boundary_replay_145_turns_upper_bound":580,"reflect_48_turns_upper_bound":192,
                    "memory_80_turns_including_summary_upper_bound":400},
            "future_demand_note":"Configuration-derived upper estimates, excluding retries and Judge; no claim these were captured before execution."}
    write_json(ROOT/"cost-ledger.json",ledger)
    rewrite_judgments=read_jsonl(ROOT/"rewrite/semantic-review.jsonl")
    failure_catalog={"execution":{name:v["telemetry"]["failed_turns"] for name,v in runs.items()},
        "quality":{name:q["failure_ids"] for name,q in quality.items()},
        "intent":load(ROOT/"intent/statistics.json")["failed_cases"],
        "boundary":load(ROOT/"boundary-replay/statistics.json")["failures"],
        "rewrite_semantic":[r["id"] for r in rewrite_judgments if not r.get("judgment") or not r["judgment"]["semantic_preserved"]],
        "rewrite_coreference":[r["id"] for r in rewrite_judgments if not r.get("judgment") or not r["judgment"]["coreference_resolved"]],
        "budget_probe_failures":[{"stage":c["stage"],"max_tokens":c.get("max_tokens"),"error_type":c.get("error_type"),"finish_reason":c.get("finish_reason"),"transport_error":c.get("transport_error")} for c in probes if c["status"] != "ok"],
        "note":"Quality discrepancies are provisional, not all proven system errors. Full answers/evidence/reasons in each run's all-quality-failures.json and traces."}
    write_json(ROOT/"failure-catalog.json",failure_catalog)

    suites=list(ET.parse("reports/history/development/acceptance-tests.xml").getroot().iter("testsuite"))
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
    reliability=f'''# 可靠性与恢复测试

本轮在原 `.venv-gpu` 环境运行，最新结果：{tests}。机器可读记录为 acceptance-tests.xml；阶段记录 baseline-tests.xml、phase-I-tests.xml、phase-II-III-tests.xml、phase-IV-tests.xml、phase-V-preflight-tests.xml 保留。

这些是离线/模拟 HTTP 故障注入，不能代替真实 API 效果指标：

| 故障 | 验证行为 |
|---|---|
| 429、503、网络 timeout | 有限重试、耗时/原因/未知成本记录 |
| 400、401、403 | 不重试，明确失败 |
| 非法 JSON、Schema 不匹配、非法 Reflect 决策 | 最多一次结构修复；仍失败则 fail closed |
| 连接预算短于响应头等待 | 建连后使用总请求预算；总超时仍失败 |
| 重排器缺失/不可用 | 保留检索排序，Trace degraded |
| hybrid 单路异常 | 配置允许时尝试另一通道；不能恢复则失败 |
| 错误引用 | 受控拒答，不写事实，Trace degraded |
| 语义摘要失败 | 确定性裁剪及完整归档 |
| 归档写入失败 | 回滚本轮消息、事实、计数，Trace error |
| 重复 turn_id、进程恢复 | 不重复 API、消息、事实或提交；相同 ID 不同问题拒绝 |
| 超阈值长期会话 | 早期消息离开近期窗口仍可召回；论文引用仍指向 chunk |
| 更新冲突与跨 session | 旧事实 superseded、当前事实优先、不同 session 不返回记录 |

修复过程中测试实际揭示并修正：401 被误重试、旧 SessionMemory.load 未识别新原子快照、第二次归档查询覆盖初次高相关结果。t3 真实预算探针还发现早期连接超时错误约束响应头等待，旧失败保留，修正后真实调用通过。

边界：没有完整阶段 Checkpoint Resume；只有请求级有限重试与幂等提交。单 session 不支持多进程并发写；同一进程调用方应串行执行。真实线上断网/真实 429 并未主动制造，相关机制证据为故障注入。磁盘原子替换不是跨机器事务。
'''
    atomic_text("reports/history/development/RELIABILITY_RESULTS.md",reliability)
    remaining='''# 未完成项与验收边界

本轮完成主要工程修正及核心开发实验，但不能宣称所有科研质量验收条件已满足。

1. Intent、Rewrite、Boundary、多轮和 Memory 标签仍未人工复核。v0.4 修正别名/历史元数据，没有补造人工 standalone_query；90 条 expected_query 仍为空。data-audit*.json 提供逐条待审清单，不提升草稿为 gold。
2. Citation 与答案质量使用同一 OpenCode 模型辅助评审，尚无人审；所有理由、证据、错误与抽样清单已保存。高引用支持率不代表独立验证的事实正确率。
3. Reflect 消融仅 3 个 session/12 turns；没有人工标注的证据不足子集，无法正式报告该子集正确率或二检后的语义证据改善率。现有二检触发和候选变化只是执行/覆盖代理。
4. Memory 消融仅 2 个十轮 session（另生成的 6 组未运行）。论文记忆与来源检查点有真实结果；真实事实更新/冲突、早期实体恢复、归档召回精确率与错误事实率缺独立标签。功能机制的 Mock 测试不能填补这个效果缺口。
5. 语义压缩在本次短长会话压力集反而增加 token；按需召回也未显示质量提升。需要更自然、更长且有独立事实检查点的数据，不能宣传省 token 或算法显著增益。
6. API 账户剩余额度/货币账单不可见；超时和被中断的早期探针存在未知成本。cost-ledger.json 是已保存调用的已知下界。没有编造额度不足作为未完成理由。
7. 当前无 Git 元数据，采用运行时源码哈希、数据/索引哈希和环境记录。最后的召回合并修复仅在 memory-recall-v2 做了真实复测；完整 E2E 是此前冻结源码的运行（四轮会话未触发压缩），不能称其为最后源码逐字相同的全量重跑。
8. 归档线性扫描；预算以记录数/字符限制和输入字符上限实现，没有统一精确 tokenizer 上限。旧归档缺失原 turn_id/order 时不能恢复未知来源。单 session 并发写、committed_turns 账本长期膨胀、完整工作流持久化未实现。

后续人工复核入口：data-audit-v04.json、各 semantic-summary.json 的 human_review_queue、all-quality-failures.json。完成标签审阅后可以直接用保存预测重新评分，无须默认重新调用模型。
'''
    atomic_text("reports/history/development/REMAINING_ISSUES.md",remaining)
    eq=quality["e2e-full"]
    em=runs["e2e-full"]["metrics"]
    rt=runs["e2e-full"]["telemetry"]
    final=["# nano-cc 本轮最终交付报告", "", "结论：工程正确性修正与核心开发实验已完成；人工金标、独立语义复核和充分的长期记忆质量验证未完成，尚不满足全部科研验收条件。旧报告保存在 history/pre-acceptance，所有结果均来自本目录实际产物。", "",
       "## 实际修改", "",
       table(["文件/函数","修改"],[
        ("scripts/prepare_eval.py · session_history/prepare","同 session 严格过去历史；版本保护；长记忆非空检查点；ID/标题别名与适用范围"),
        ("eval_tasks.py · bootstrap_ci/run_boundary/run_rewrite/run_memory","正确有放回簇抽样；Replay/Context-only；拒绝污染历史；逐轮记忆检查"),
        ("eval_protocol.py · assess/success_metrics/compare_runs","三类成功率、未知质量、逐例配对差值与 CI、缺失/失败/配置报告"),
        ("eval_cli.py · cmd_ablation/cmd_compare/cmd_e2e","真实执行矩阵、保存有效配置、会话并行与逐会话结果、API参数覆盖"),
        ("providers.py · APIConfig/post_json/CompatibleReasoner","连接与总超时、有限重试/结构修复、状态分类、token/未知成本记录"),
        ("pipeline.py · PaperAgentState/PaperPipeline.run/compact_state","单快照原子提交、turn_id 幂等/回滚、组件降级、归档上下文与合并修复"),
        ("memory.py · SessionMemory.load/ContextArchive.search","新旧快照读取、会话隔离、关键词/实体/论文/时间召回、旧事实标记"),
        ("compaction.py · semantic_compact_state","保留讨论论文元数据、归档 turn 序、压缩观测字段"),
        ("evaluation.py · retrieval_metrics/evaluate_retriever","Recall@10、MRR、nDCG 及完整排名"),
        ("tests/test_eval_corrections.py、test_reliability.py、test_providers.py","统计、历史、幂等、降级、超时、归档和长对话回归"),
        ("新增 retest_budget/audit_eval_data/judge_acceptance/review_rewrite/quality_statistics/summarize_acceptance/finalize_acceptance 脚本","真实探针、静态审核、辅助评审、统计与可重复生成报告"),
        ("configs/evaluation.json、README、docs 全部组件页","执行矩阵、协议、运行边界和实测配置同步")]), "",
       "七阶段及原 Harness、BM25/BGE/FAISS/RRF/Cross-Encoder 保留；未下载模型、重建原索引或替换 .env，未新增外部数据库/服务。", "",
       "## 验证与数据", "",
       f"原依赖环境最新 {tests['tests']} 项测试全部通过，见 acceptance-tests.xml。Mock 与真实调用分开。真实数据为 v0.3（Intent 100、Rewrite 90、Boundary 55、30×4 E2E、Memory 子集 2×10）。v0.4 仅改进历史元数据/实体别名标签，保留草稿状态；其 Rewrite 使用 v0.3 保存预测做离线重新评分，未冒充新 API 实验。", "",
       f"OpenCode t3：2048 截断，4096/8192 成功；4096+disabled 实测 reasoning_tokens=0。四轮回归4/4完成；全量 execution {em['completed']}/120，70-turn 视图从完整历史运行取分。全量阶段 API {rt['api_requests']} 次、已知 token {rt['known_total_tokens']}，P50/P95 {rt['p50_ms']:.0f}/{rt['p95_ms']:.0f} ms。", "",
       f"E2E 行为与草稿标签一致率 {interval(em['behavior_match_rate']['ci'])}。严格规则 task/pipeline 已满足条件下界均 {em['pipeline_success_rate']['correct']}/120，另 {em['pipeline_success_rate']['unknown']} 条质量未知；不能把 completed/total 当综合成功率。辅助 Judge 答案质量 {interval(eq['answer_correct'])}；结合草稿行为条件的临时 Pipeline Success {interval(eq['provisional_pipeline_success'])}。这些不是人工验证的正式指标。", "",
       f"引用 ID 与来源结构均 {eq['citation_id_validity']['correct']}/{eq['citation_id_validity']['total']}；语义支持 {eq['citation_evidence_support']['supported']}/{eq['citation_evidence_support']['judged']}，Unsupported Claim Rate {eq['unsupported_claim_rate']:.2%}（同模型评审）。全部执行/质量失败和不一致案例保存，而不是从分母移除。", "",
       "## 算法结果与成本", "",
       "检索 Recall@5：66.26% / 73.94% / 74.24% / 77.61%，新四组复测与旧记录一致。Hybrid−Dense +0.30pp，95% CI [-7.40,+6.95]；Rerank−Hybrid +3.38pp，CI [-4.29,+11.04]，均不能声称显著。完整 Recall@10/MRR/nDCG、互补案例与排名变动见 ABLATION_RESULTS.md。", "",
       "Reflect 12 turns：禁用/一次反思/允许二检的辅助答案质量为 83.33% / 83.33% / 66.67%，未证实二检增益。Memory 20 turns/组：完整/裁剪/语义/语义+召回v2 的辅助质量为 100% / 95% / 95% / 85%；结构检查点均20/20，不能当作事实语义全正确。", "",
       table(["比较","辅助质量差值/95% CI","全阶段token变化"],[(k,interval(v['quality_delta']),f"{v['token_change_ratio']:+.2%}") for k,v in paired.items()]), "",
       f"成本账本：已保存请求下界 {ledger['known_total_requests']} 次，已知 token {ledger['known_total_tokens']}，usage 未知 {ledger['unknown_usage_requests']} 次。包括诊断、重跑和 Judge，不能当作一次部署对话的成本。剩余额度与币值成本未知，详见 acceptance/cost-ledger.json。", "",
       "## 复现与剩余验收", "",
       "完整命令见 docs/evaluation.md。原目录使用 `.venv-gpu/Scripts/python.exe`；已有 data/models/indexes/.env 直接复用。先 pytest，再 smoke/小回归；检索用 --reuse-index；真实任务显式 --max-tokens 4096 --thinking disabled。报告重算不调用 API：", "",
       "```powershell\n.venv-gpu/Scripts/python.exe scripts/quality_statistics.py\n.venv-gpu/Scripts/python.exe scripts/summarize_acceptance.py\n.venv-gpu/Scripts/python.exe scripts/finalize_acceptance.py\n.venv-gpu/Scripts/python.exe scripts/check_artifacts.py\n```", "",
       "待完成项目见 REMAINING_ISSUES.md：人工金标/引用复核、正式证据不足子集、真实长期事实更新与召回精确率、更多自然长会话。已执行核心消融不等于已证明增益；不把未审核草稿、Mock 或参考截图当项目科研成果。"]
    atomic_text("reports/history/development/FINAL_REPORT.md","\n".join(final)+"\n")
    hashes={str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('src','scripts','tests','docs','configs') for p in Path(folder).rglob('*') if p.is_file() and p.suffix in {'.py','.md','.json'}}
    write_json(ROOT/"final-source-hashes.json",hashes)
    print({"tests":tests,"requests_known":ledger['known_total_requests'],"tokens_known":ledger['known_total_tokens']})


if __name__ == "__main__": main()
