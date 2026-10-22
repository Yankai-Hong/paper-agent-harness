"""Render saved retrieval outputs; paired bootstrap clusters by source paper."""
import argparse
import json
from pathlib import Path

import numpy as np
from paper_agent.storage import read_jsonl, write_json, atomic_text, digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="reports/qasper-suite")
    parser.add_argument("--queries", default="data/qasper/retrieval.jsonl")
    args = parser.parse_args()
    run = Path(args.run)
    queries = read_jsonl(args.queries)
    modes = ["bm25", "dense", "hybrid", "rerank"]
    metrics = {mode: json.loads((run/mode/"metrics.json").read_text()) for mode in modes}
    if any(m["queries_hash"] != digest(queries) for m in metrics.values()):
        raise ValueError("Query manifest mismatch")
    if len({m["corpus_hash"] for m in metrics.values()}) != 1:
        raise ValueError("Cannot compare runs over different corpora")
    papers = sorted({q["source_paper_id"] for q in queries})
    clusters = {p: [i for i, q in enumerate(queries) if q["source_paper_id"] == p] for p in papers}
    values = {}
    for mode in modes:
        scores = {p["query_id"]: p["recall"] for p in metrics[mode]["per_query"]}
        values[mode] = np.array([scores[q["query_id"]] for q in queries])
    rng = np.random.default_rng(42)
    samples = [np.concatenate([clusters[p] for p in rng.choice(papers, len(papers), replace=True)]) for _ in range(2000)]
    deltas = {}
    for name, new, old in [("hybrid_minus_dense", "hybrid", "dense"), ("rerank_minus_hybrid", "rerank", "hybrid")]:
        differences = values[new]-values[old]
        interval = np.quantile([differences[ids].mean()*100 for ids in samples], [.025, .975])
        deltas[name] = {"percentage_points": float(differences.mean()*100),
                        "bootstrap_95_percentile_interval_pp": interval.tolist()}
    write_json(run/"paired_comparison.json", {"seed": 42, "bootstrap_samples": 2000,
               "clusters": len(papers), "cluster_unit": "source paper", "deltas": deltas})
    failures = [{"query_id": q["query_id"], "query": q["query"], "paper_id": q["source_paper_id"],
                 "gold_units": q["gold_units"], "rerank_recall": float(values["rerank"][i])}
                for i, q in enumerate(queries) if values["rerank"][i] < 1]
    write_json(run/"incomplete_retrieval_cases.json", failures)
    lines = ["# 真实 GPU 检索对照结果", "", "数据：QASPER v0.3，100 篇全文、100 条官方文本证据问题；已知目标论文范围。",
             "四组使用同一语料、问题、BGE 分词与 30 候选预算；未按测试结果调参；无付费 API 调用。", "",
             "| 配置 | Recall@5 | Hit@5 | 失败数 | 100题检索耗时 |", "|---|---:|---:|---:|---:|"]
    for mode in modes:
        m = metrics[mode]
        lines.append(f"| {mode} | {m['Recall@5']:.2%} | {m['Hit@5']:.2%} | {m['errors']} | {m['elapsed_seconds']:.2f}s |")
    lines += ["", "耗时不包含模型加载和建索引；单机单次测量，包含每题查询编码；不作为吞吐保证。", ""]
    for name, value in deltas.items():
        lo, hi = value["bootstrap_95_percentile_interval_pp"]
        lines.append(f"- {name}: {value['percentage_points']:+.2f} 个百分点；按论文聚类的配对 bootstrap 95% 百分位区间 [{lo:+.2f}, {hi:+.2f}]。")
    lines += ["", "限制：英文单轮、已知论文、段落覆盖代理指标；任一滑窗命中计作对应段落被召回，不保证完整答案证据覆盖。",
              "原始全库压力结果单列在 ../qasper-bm25-bgechunks，不与本表混比。没有自建多轮问答效果，也不等同于截图数值复现。",
              "引用是否支持事实、答案是否正确需另行独立评价；本次仅评价检索。",
              "", "可核验文件：metrics.json、predictions.jsonl、paired_comparison.json、run_config.json、environment.json 和 reranker_provenance.json。"]
    atomic_text(run/"RESULTS.md", "\n".join(lines)+"\n")
    print("Report generated", flush=True)
