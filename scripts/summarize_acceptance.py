"""Recompute acceptance reports from saved outputs; no network calls."""
import json
import hashlib
from pathlib import Path
import numpy as np
from paper_agent.storage import read_jsonl, write_json, atomic_text
from paper_agent.eval_tasks import bootstrap_ci
from paper_agent.eval_protocol import compare_runs, success_metrics

ROOT = Path("reports/history/development/acceptance")


def load(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def telemetry(root):
    traces = [load(p) for p in (root/"traces").rglob("*.json")]
    calls = [c for t in traces for c in t.get("api_calls", [])]
    if (root/"api_calls.json").exists(): calls += load(root/"api_calls.json")
    latencies = [sum(s.get("elapsed_ms", 0) for s in t["stages"]) for t in traces]
    if not latencies: latencies = [c["elapsed_ms"] for c in calls]
    errors = [{"session_id":t["session_id"], "turn_id":t["turn_id"],
               "stages":[{"stage":s["stage"],"error":s.get("error")} for s in t["stages"] if s["status"] == "error"]}
              for t in traces if t["status"] == "error"]
    token_total = sum(sum(c.get("usage", {}).values()) for c in calls)
    return {"trace_turns":len(traces), "api_requests":len(calls), "api_failed_attempts":sum(c["status"] != "ok" for c in calls),
            "known_total_tokens":token_total, "unknown_usage_calls":sum(not c.get("usage_complete") for c in calls),
            "mean_known_tokens_per_trace_turn":token_total/len(traces) if traces else None,
            "p50_ms":float(np.percentile(latencies,50)) if latencies else None,
            "p95_ms":float(np.percentile(latencies,95)) if latencies else None,
            "mean_ms":float(np.mean(latencies)) if latencies else None,
            "degraded_turns":sum(t.get("degraded",False) for t in traces), "failed_turns":errors,
            "recall_turns":sum(bool(t.get("archive_recall")) for t in traces),
            "recalled_records":sum(len(t.get("archive_recall",[])) for t in traces),
            "compressed_turns":sum(any(s.get("compaction",{}).get("compressed") for s in t["stages"]) for t in traces),
            "retrieve_again_turns":sum(sum(s["stage"] == "Retrieve" for s in t["stages"]) > 1 for t in traces)}


def percent(value):
    return "待完成/未知" if value is None else f"{value:.2%}"


def interval(value):
    if not value: return "未知"
    lo,hi=value["bootstrap_95_percentile_interval"]
    return f"{value['mean']:.2%} [{lo:.2%}, {hi:.2%}]"


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"|".join(["---"]*len(headers))+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])


def summarize():
    roots = [p.parent for p in ROOT.rglob("metrics.json") if "traces" not in p.parts and "retrieval" not in str(p)]
    runs = {p.relative_to(ROOT).as_posix(): {"metrics":load(p/"metrics.json"), "telemetry":telemetry(p),
             "semantic":load(p/"semantic-summary.json")} for p in roots}
    queries = read_jsonl("data/qasper/retrieval.jsonl")
    clusters = [q["source_paper_id"] for q in queries]
    retrieval = {}
    for mode in ("bm25","dense","hybrid","rerank"):
        root=ROOT/"retrieval-v2"/mode
        m=load(root/"metrics.json")
        if not m: continue
        by_id={r["query_id"]:r for r in m["per_query"]}
        retrieval[mode] = {"metrics":m, "recall_ci":bootstrap_ci([by_id[q["query_id"]]["recall"] for q in queries],clusters)}
    paired=load(ROOT/"retrieval-v2/paired_comparison.json",{})
    disagreements=[]
    if len(retrieval)==4:
        by_mode={mode:{r["query_id"]:r for r in obj["metrics"]["per_query"]} for mode,obj in retrieval.items()}
        for q in queries:
            values={mode:by_mode[mode][q["query_id"]]["recall"] for mode in by_mode}
            if (values["dense"]>0) != (values["bm25"]>0):
                disagreements.append({"query_id":q["query_id"],"question":q["query"], "recall":values})
    write_json(ROOT/"retrieval-v2/complementary-cases.json",disagreements)

    # Alias-aware diagnostics preserve all predictions; no additional model calls or label promotion.
    rewrite_rows=read_jsonl("data/eval/v0.3/query_rewrite.jsonl")
    rewrite_predictions={r["id"]:r for r in read_jsonl(ROOT/"rewrite/per_case.jsonl")}
    titles={r["paper_id"]:r["title"] for r in read_jsonl("data/qasper/papers.jsonl")}
    aliases=[]
    for row in rewrite_rows:
        pred=rewrite_predictions.get(row["id"],{})
        text=json.dumps([pred.get("standalone_query",""), pred.get("resolved_entities",{}), pred.get("paper_ids",[])],ensure_ascii=False).casefold()
        def normalize(x): return " ".join(__import__('re').findall(r"\w+",x.casefold()))
        hits=[p for p in row["required_entities"] if p.casefold() in text or normalize(titles[p]) in normalize(text)]
        aliases.append({"id":row["id"],"session_id":row["session_id"],"turn":row["turn"],
                        "score":len(hits)/len(row["required_entities"]) if row["required_entities"] else None,
                        "matched_papers":hits, "applicable":row["turn"] in {2,3}})
    applicable=[r for r in aliases if r["applicable"] and r["score"] is not None]
    alias_metric={"all_90_diagnostic":bootstrap_ci([r["score"] for r in aliases],[r["session_id"] for r in aliases]),
                  "in_scope_followups_60":bootstrap_ci([r["score"] for r in applicable],[r["session_id"] for r in applicable]),
                  "note":"Post-audit ID/title alias scoring. All 90 outcomes retained. Turn 4 boundary questions not in the 60-turn entity-specific denominator. Labels remain provisional."}
    write_json(ROOT/"rewrite/entity-alias-metrics.json",alias_metric)
    write_json(ROOT/"rewrite/entity-alias-cases.json",aliases)

    # Confirmed success and provisional model quality remain separate.
    full_path=ROOT/"e2e-full/predictions.jsonl"
    e2e70={}
    if full_path.exists():
        ids={r["turn_id"] for r in read_jsonl("data/dialogues-draft/e2e-view.jsonl")}
        subset=[p for p in read_jsonl(full_path) if p["turn_id"] in ids]
        e2e70={"n":len(subset), **success_metrics(subset), "source":"scored from full 120-turn replay; no histories skipped"}
        write_json(ROOT/"e2e-70-metrics.json",e2e70)

    boundary_rows=read_jsonl("data/eval/v0.3/boundary.jsonl")
    boundary_preds={r["id"]:r for r in read_jsonl(ROOT/"boundary-replay/per_case.jsonl")}
    boundary_ci=bootstrap_ci([float(boundary_preds.get(r["id"],{}).get("decision")==r["decision"]) for r in boundary_rows],
                              [r["session_id"] for r in boundary_rows])
    bm=runs["boundary-replay"]["metrics"]
    precision=bm["refusal_precision"]["rate"] or 0
    recall=bm["refusal_recall"]["rate"] or 0
    boundary_extra={"accuracy_ci":boundary_ci,"refusal_f1":2*precision*recall/(precision+recall) if precision+recall else 0,
                    "out_of_scope_answer_rate":sum(boundary_preds.get(r["id"],{}).get("decision")=="answer" for r in boundary_rows if r["decision"]=="refuse")/sum(r["decision"]=="refuse" for r in boundary_rows),
                    "failures":[r["id"] for r in boundary_rows if boundary_preds.get(r["id"],{}).get("decision") != r["decision"]]}
    write_json(ROOT/"boundary-replay/statistics.json",boundary_extra)

    intent_rows=read_jsonl("data/eval/v0.3/intent.jsonl")
    ip={r["id"]:r.get("intent") for r in read_jsonl(ROOT/"intent/per_case.jsonl")}
    intent_ci=bootstrap_ci([float(ip.get(r["id"])==r["intent"]) for r in intent_rows],[r["session_id"] for r in intent_rows])
    write_json(ROOT/"intent/statistics.json",{"accuracy_ci":intent_ci,"failed_cases":[r["id"] for r in intent_rows if ip.get(r["id"])!=r["intent"]]})

    comparisons={}
    for a,b,metric in [("reflect-disabled","reflect-retrieve-again","behavior_match"),
                        ("reflect-enabled","reflect-retrieve-again","behavior_match"),
                        ("memory-full","memory-bounded","checkpoint_pass"),
                        ("memory-bounded","memory-semantic","checkpoint_pass")]:
        pa,pb=ROOT/"ablations"/a, ROOT/"ablations"/b
        if (pa/"metrics.json").exists() and (pb/"metrics.json").exists():
            comparisons[b+"-minus-"+a]=compare_runs(pa,pb,metric)
    if (ROOT/"memory-recall-v2/metrics.json").exists():
        comparisons["recall-v2-minus-semantic"]=compare_runs(ROOT/"ablations/memory-semantic",ROOT/"memory-recall-v2","checkpoint_pass")
    write_json(ROOT/"paired-comparisons.json",comparisons)
    summary={"runs":runs,"retrieval":retrieval,"paired_retrieval":paired,"boundary":boundary_extra,
             "intent_accuracy_ci":intent_ci,"entity_alias":alias_metric,"e2e70":e2e70,"comparisons":comparisons}
    write_json(ROOT/"summary.json",summary)

    live=["# 本轮真实 OpenCode 评测", "", "原 provider/.env 未替换；真实调用与离线故障注入分开。旧报告保留于 history/pre-acceptance。开发批次使用 deepseek-v4.1-flash、4096、thinking=disabled；所有草稿/模型 Judge 指标均 provisional。", "",
          "t3 固定输入预算对照见 acceptance/budget-v2/results.json：2048 截断；4096/8192 成功；disabled 实测 reasoning_tokens=0。此前连接超时实现缺陷造成的失败保留在 budget/；中断中的请求成本可能未知。四轮端到端 smoke 完成 4/4。", "",
          f"Intent 100 条：Accuracy {interval(intent_ci)}，Macro-F1 {runs['intent']['metrics']['macro_f1']:.4f}；完整混淆矩阵和 15 个不一致案例见 intent/。",
          f"Rewrite 90 条：原 ID 字面匹配指标 {percent(runs['rewrite']['metrics']['entity_retention_rate'])} 不等同于实体丢失。新增别名诊断全部 90 条 {interval(alias_metric['all_90_diagnostic'])}；实体适用的 t2/t3 共 60 条 {interval(alias_metric['in_scope_followups_60'])}。保留原始结果及口径变更，不改样本或补造 expected_query。语义/指代与检索增益见 rewrite/semantic-summary.json，均为辅助或来源证据代理。",
          f"Boundary Replay 55 条：一致率 {interval(boundary_ci)}；拒答 Precision {percent(precision)}、Recall {percent(recall)}、F1 {boundary_extra['refusal_f1']:.4f}；5 个不一致案例 {boundary_extra['failures']}。Context-only 没有混入本表。", "",
          "## 运行与用量（阶段调用含重试，不含单列辅助 Judge）", "",
          table(["运行","执行轮/调用","已知总 token","未知 usage 调用","P50/P95 ms","降级轮"],
                [(name,f"{v['telemetry']['trace_turns']}/{v['telemetry']['api_requests']}",v['telemetry']['known_total_tokens'],v['telemetry']['unknown_usage_calls'],
                  f"{v['telemetry']['p50_ms']:.0f}/{v['telemetry']['p95_ms']:.0f}" if v['telemetry']['p50_ms'] is not None else "未知",v['telemetry']['degraded_turns'])
                 for name,v in sorted(runs.items())]), "",
          "完整 120 turns 和从其计分的 70-turn 视图分别保存在 e2e-full/ 与 e2e-70-metrics.json。execution/task/pipeline 含未知质量的下界不可混用。Citation ID Validity 与 Source Consistency 是结构检查；逐 claim 支持和 Unsupported Claim Rate 见 semantic-summary.json，是同模型评审且待人工抽查。未评审/失败案例不会被悄悄算作正确。"]
    for name in ["e2e-full", "e2e-smoke"]:
        if name in runs:
            m=runs[name]["metrics"]
            live += ["",f"{name}：execution {m.get('execution_success_rate')}；task {m.get('task_success_rate')}；pipeline {m.get('pipeline_success_rate')}。"]
    atomic_text("reports/history/development/LIVE_EVAL_RESULTS.md","\n".join(live)+"\n")

    retrieval_table=table(["配置","Recall@5/95% CI","Recall@10","MRR@5","nDCG@5","失败/总数"],
       [(mode,interval(v["recall_ci"]),percent(v["metrics"].get("Recall@10")),f"{v['metrics']['MRR@5']:.4f}",f"{v['metrics']['nDCG@5']:.4f}",f"{v['metrics']['errors']}/100") for mode,v in retrieval.items()])
    ablation=["# 检索与 Reflect 消融", "", "本轮执行结果，不使用参考截图预测值。固定输入/来源哈希见每组 run_config.json。", "", retrieval_table,"",
              "配对区间与全量逐题排名见 acceptance/retrieval-v2。Dense/BM25 互补案例见 complementary-cases.json。RRF 两路排名重叠、已知论文范围较窄，因此融合相对 dense 的平均收益较小；这是对保存排名与任务设置的解释，不是显著因果结论。重排改善均值但区间跨零，不能声称显著。", "", "## Reflect 开发对照", "",
              "12 turns、3 sessions、相同 BM25、模型与预算。enabled 与 no-second 都只允许一次检索，是重复对照，采样波动不能当作算法差异。其他组为 disabled 与允许二次检索。", "",
              table(["配置","完成/总数","行为一致率","二检轮","API次数","已知token","辅助质量/95% CI"],
                [(name.split('/')[-1],f"{v['metrics'].get('completed')}/{v['metrics'].get('turns')}",percent(v['metrics'].get('behavior_match_rate',{}).get('rate')),
                  v['telemetry']['retrieve_again_turns'],v['telemetry']['api_requests'],v['telemetry']['known_total_tokens'],interval((v.get('semantic') or {}).get('answer_correct_lower_bound')))
                 for name,v in sorted(runs.items()) if name.startswith('ablations/reflect')]), "",
              "逐例决策、候选证据、答案、错误与 API usage 在 traces/ 与 predictions.jsonl。独立预标的证据不足子集尚缺人工审核；不能用模型自己的 retrieve_again 标签当该子集 gold。二检触发不等于证据质量改善。配对行为差异见 acceptance/paired-comparisons.json，语义 Judge 不升级为人工标签。"]
    atomic_text("reports/history/development/ABLATION_RESULTS.md","\n".join(ablation)+"\n")
    memory=["# 记忆压缩与召回评测", "", "三层记忆保留。v0.3 有 8×10 轮长会话；真实消融固定前 2 个 session（20 turns/组），其余未执行，不能声称 80 个不同会话问题均已实测。检查点是讨论论文、事实证据来源，非人工事实文本金标。", "",
            table(["组","检查点通过/总数","压缩轮","召回轮/记录","API次数","已知token","P50/P95 ms","Judge质量/95% CI"],
              [(name,v['metrics'].get('checkpoint_accuracy'),v['telemetry']['compressed_turns'],f"{v['telemetry']['recall_turns']}/{v['telemetry']['recalled_records']}",
                v['telemetry']['api_requests'],v['telemetry']['known_total_tokens'],f"{v['telemetry']['p50_ms']:.0f}/{v['telemetry']['p95_ms']:.0f}" if v['telemetry']['p50_ms'] else "未知",
                interval((v.get('semantic') or {}).get('answer_correct_lower_bound')))
               for name,v in sorted(runs.items()) if 'memory-' in name]), "",
            "memory-recall-v2 是修复两次召回覆盖问题后的独立重跑，旧 memory-recall 保留；应使用 v2 与 semantic 对照。压缩 tokens 是实际全阶段 usage（包括摘要/重试），不能只用字符估计声称成本节省。token 比较和 checkpoint 配对 CI 见 acceptance/summary.json、paired-comparisons.json。", "",
            "已验证的机制：长会话早期消息移出近期窗口后可以召回；同 session 隔离；同 key 新事实优先且旧记录标记 superseded；重复 turn 与进程恢复不重复提交；归档写失败不部分提交。相关证据来自 test_reliability.py，属于 Mock/离线功能测试。", "",
            "尚未独立验证：真实模型对人工标注事实更新/冲突、归档召回精确率、早期实体指代准确率、错误事实率的完整金标质量。此次长集重复问题较多，工作 facts 可能未超 40，不能用其 checkpoint 满分证明任意长对话记忆可靠。"]
    atomic_text("reports/history/development/MEMORY_EVAL_RESULTS.md","\n".join(memory)+"\n")
    print("Reports regenerated from saved results",flush=True)


if __name__ == "__main__": summarize()
