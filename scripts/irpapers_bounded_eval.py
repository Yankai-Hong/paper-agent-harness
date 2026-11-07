"""Freeze a bounded Chroma page depth on development, then evaluate all cases."""
import argparse
import json
import statistics
import time
from pathlib import Path

from irpapers_optimize import read_frozen, sha


def rows(path, dev_only=False):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if dev_only and '"partition": "dev"' not in line:
                continue
            if line.strip():
                yield json.loads(line)


def metrics(rows):
    n = len(rows)
    def hit(r, k): return r["gold_rank"] is not None and r["gold_rank"] <= k
    return {"n": n, "hit1": sum(hit(r, 1) for r in rows) / n, "hit5": sum(hit(r, 5) for r in rows) / n,
            "hit20": sum(hit(r, 20) for r in rows) / n,
            "mrr_at_20": sum(1 / r["gold_rank"] if r["gold_rank"] is not None else 0 for r in rows) / n,
            "top5_exact_overlap_mean": sum(r["top5_overlap"] for r in rows) / (5 * n),
            "satisfied": sum(r["satisfied"] for r in rows),
            "hnsw": sum(r["mode"] == "chroma_hnsw_cosine" for r in rows)}


def run(args, depths, dev_only):
    import numpy as np
    import psutil
    import torch
    from paper_agent.bounded_search import search_grouped
    from paper_agent.retrieval import BGEEncoder
    from paper_agent.vector_store import ChromaVectorStore
    root = Path(args.baseline)
    baseline = list(rows(root / "text-results.jsonl", dev_only))
    chunks = list(rows(Path(args.index) / "chunks.jsonl"))
    meta = json.loads((Path(args.index) / "metadata.json").read_text(encoding="utf-8"))
    source = {c["chunk_id"]: (c["paper_id"], c["page_id"]) for c in chunks}
    process = psutil.Process()
    before = process.memory_info().rss
    t = time.perf_counter()
    store = ChromaVectorStore(Path(args.index) / "vectors", meta["vector_contract"])
    load_seconds = time.perf_counter() - t
    after_store = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    encoder = BGEEncoder("models/bge-m3", "cuda:0", 2, 768)
    t = time.perf_counter()
    vectors = encoder.encode([r["question"] for r in baseline])
    encode_seconds = time.perf_counter() - t
    results = []
    for row, vector in zip(baseline, vectors):
        if row["dense"]["status"] != "ok":
            raise ValueError("Frozen dense query failed")
        for depth in depths:
            start = time.perf_counter()
            result = search_grouped(store, vector, source, limit=20, initial_depth=depth, max_depth=depth)
            elapsed = time.perf_counter() - start
            pages = [x["page_id"] for x in result["results"]]
            gold = row["gold_page"]
            rank = pages.index(gold) + 1 if row["gold_status"] == "consistent" and gold in pages else None
            results.append({"case_id": row["case_id"], "group": row["group"], "partition": row["partition"],
                            "gold_rank": rank, "exact_gold_rank": row["dense"]["gold_rank"],
                            "pages": pages, "exact_top20": row["dense"]["pages"][:20], "depth": depth,
                            "top5_overlap": len(set(pages[:5]) & set(row["dense"]["pages"][:5])),
                            "satisfied": result["satisfied"], "mode": result["search_mode"],
                            "search_seconds": elapsed})
    search_rss = process.memory_info().rss
    # Time exact scoring separately after the bounded measurements, so the
    # exact vector cache does not contaminate bounded-search RAM measurements.
    exact_seconds = []
    for vector in vectors[:10]:
        start = time.perf_counter()
        store.search(vector, store.manifest["count"])
        exact_seconds.append(time.perf_counter() - start)
    resources = {"store_open_seconds": load_seconds, "rss_before": before, "rss_after_store": after_store,
                 "rss_after_bounded": search_rss, "rss_after_exact": process.memory_info().rss,
                 "bge_encoding_seconds": encode_seconds, "queries_encoded": len(vectors),
                 "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "exact_seconds_first": exact_seconds[0], "exact_seconds_warm_median": statistics.median(exact_seconds[1:]),
                 "exact_sample_size": len(exact_seconds), "exact_mode": store.last_search_mode}
    return results, resources


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def choose(args):
    root = Path(args.baseline)
    frozen = read_frozen(root)
    result, resources = run(args, [100, 300, 600], True)
    options = []
    for depth in (100, 300, 600):
        subset = [r for r in result if r["depth"] == depth]
        groups = {g: metrics([r for r in subset if r["group"] == g]) for g in ("standard", "visual")}
        exact_hits = {g: sum(r["exact_gold_rank"] is not None and r["exact_gold_rank"] <= 5 for r in subset if r["group"] == g)
                      for g in groups}
        acceptable = all(round(groups[g]["hit5"] * groups[g]["n"]) >= exact_hits[g] and groups[g]["top5_exact_overlap_mean"] >= .99
                         for g in groups) and all(r["satisfied"] and r["mode"] == "chroma_hnsw_cosine" for r in subset)
        options.append({"depth": depth, "metrics": groups, "exact_hit5_counts": exact_hits, "eligible": acceptable,
                        "median_seconds": statistics.median(r["search_seconds"] for r in subset)})
    eligible = [x for x in options if x["eligible"]]
    winner = min(eligible, key=lambda x: x["depth"]) if eligible else None
    plan = {"status": "frozen_before_test_evaluation", "baseline_freeze_sha256": sha(root / "BASELINE-FROZEN.json"),
            "baseline_files": frozen["files"], "script_sha256": sha(__file__),
            "bounded_code_sha256": sha("src/paper_agent/bounded_search.py"), "options": options, "winner": winner,
            "selection": "smallest depth with >=99% mean top5 overlap and no Hit@5 loss in either development group",
            "strict_blind_test": False, "resources": resources}
    out = Path(args.output)
    write_new(out, plan)
    detail = Path(str(out) + ".development.jsonl")
    with detail.open("x", encoding="utf-8") as stream:
        for r in result:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"plan_sha256": sha(out), "options": options, "resources": resources}, ensure_ascii=False))


def evaluate(args):
    root = Path(args.baseline)
    frozen = read_frozen(root)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if (plan["status"] != "frozen_before_test_evaluation" or plan["baseline_freeze_sha256"] != sha(root / "BASELINE-FROZEN.json")
            or plan["baseline_files"] != frozen["files"] or plan["script_sha256"] != sha(__file__)
            or plan["bounded_code_sha256"] != sha("src/paper_agent/bounded_search.py") or not plan["winner"]):
        raise ValueError("Frozen bounded plan changed or no eligible candidate")
    result, resources = run(args, [plan["winner"]["depth"]], False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        for r in result:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {}
    for group in ("standard", "visual"):
        for partition in ("dev", "test"):
            subset = [r for r in result if r["group"] == group and r["partition"] == partition]
            data = metrics(subset)
            for k in (1, 5, 20):
                hit = lambda r, field: r[field] is not None and r[field] <= k
                data[f"hit{k}_improved_vs_exact"] = [r["case_id"] for r in subset if hit(r, "gold_rank") and not hit(r, "exact_gold_rank")]
                data[f"hit{k}_regressed_vs_exact"] = [r["case_id"] for r in subset if hit(r, "exact_gold_rank") and not hit(r, "gold_rank")]
            data["median_seconds"] = statistics.median(r["search_seconds"] for r in subset)
            summary[group + "_" + partition] = data
    data = {"plan_sha256": sha(args.plan), "results_sha256": sha(out), "metrics": summary, "resources": resources,
            "mrr_scope": "MRR@20 only; missing later pages are censored, not full MRR", "strict_blind_test": False}
    write_new(str(out) + ".summary.json", data)
    print(json.dumps(data, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("choose", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--baseline", required=True)
        p.add_argument("--index", required=True)
        p.add_argument("--output", required=True)
        if command == "evaluate":
            p.add_argument("--plan", required=True)
    args = parser.parse_args()
    (choose if args.command == "choose" else evaluate)(args)


if __name__ == "__main__":
    main()
