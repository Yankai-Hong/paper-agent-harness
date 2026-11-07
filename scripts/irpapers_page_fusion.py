"""Freeze development-only BM25/BGE page fusion, then evaluate preserved splits."""
import argparse
import json
import time
from pathlib import Path

from irpapers_optimize import load_dev, rank_gold, read_frozen, score, sha
from paper_agent.page_fusion import fuse_page_rankings


def choose(args):
    root = Path(args.baseline)
    frozen = read_frozen(root)
    rows = list(load_dev(root / "text-results.jsonl"))
    if any(any(r[x]["status"] != "ok" for x in ("bm25", "dense", "hybrid")) for r in rows):
        raise ValueError("Baseline contains a failed branch")
    old = {group: score([r["hybrid"]["gold_rank"] for r in rows if r["group"] == group])
           for group in ("standard", "visual")}
    old_all = score([r["hybrid"]["gold_rank"] for r in rows])
    options = []
    for k in (10, 30, 60, 120):
        for weight in (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 4):
            groups = {"standard": [], "visual": []}
            for r in rows:
                pages = fuse_page_rankings([r["bm25"]["pages"], r["dense"]["pages"]], [1, weight], k=k)
                groups[r["group"]].append(rank_gold(pages, r))
            stats = {g: score(rs) for g, rs in groups.items()}
            stats["all"] = score(groups["standard"] + groups["visual"])
            options.append({"k": k, "dense_weight": weight, "metrics": stats,
                            "eligible": all(stats[g]["hit5"] >= old[g]["hit5"] for g in groups)})
    eligible = [o for o in options if o["eligible"] and o["metrics"]["all"]["mrr"] > old_all["mrr"]]
    winner = max(eligible, key=lambda o: (o["metrics"]["all"]["mrr"], -o["k"], -o["dense_weight"])) if eligible else None
    plan = {"status": "frozen_before_test_evaluation", "baseline_freeze_sha256": sha(root / "BASELINE-FROZEN.json"),
            "baseline_files": frozen["files"], "script_sha256": sha(__file__),
            "fusion_code_sha256": sha("src/paper_agent/page_fusion.py"), "development_cases": len(rows),
            "selection": "max combined dev MRR; require improvement versus text baseline and no group Hit@5 decrease",
            "strict_blind_test": False, "reason": "previous baseline test aggregates already inspected",
            "baseline_dev": {**old, "all": old_all}, "options": options, "winner": winner}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"sha256": sha(out), "winner": winner}, ensure_ascii=False))


def evaluate(args):
    root = Path(args.baseline)
    frozen = read_frozen(root)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if (plan["status"] != "frozen_before_test_evaluation" or plan["baseline_freeze_sha256"] != sha(root / "BASELINE-FROZEN.json")
            or plan["baseline_files"] != frozen["files"] or plan["script_sha256"] != sha(__file__)
            or plan["fusion_code_sha256"] != sha("src/paper_agent/page_fusion.py") or not plan["winner"]):
        raise ValueError("Selection, baseline, or implementation mismatch")
    with (root / "text-results.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    winner = plan["winner"]
    outrows = []
    for row in rows:
        if any(row[x]["status"] != "ok" for x in ("bm25", "dense", "hybrid")):
            raise ValueError("Baseline contains a failed branch")
        start = time.perf_counter()
        pages = fuse_page_rankings([row["bm25"]["pages"], row["dense"]["pages"]],
                                   [1, winner["dense_weight"]], k=winner["k"])
        elapsed = time.perf_counter() - start
        outrows.append({"case_id": row["case_id"], "partition": row["partition"], "group": row["group"],
                        "gold_page": row["gold_page"], "gold_status": row["gold_status"],
                        "old_rank": row["hybrid"]["gold_rank"], "new_rank": rank_gold(pages, row),
                        "bm25_rank": row["bm25"]["gold_rank"], "new_top20": pages[:20], "fusion_seconds": elapsed})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in outrows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = {}
    for group in ("standard", "visual"):
        for part in ("dev", "test"):
            subset = [r for r in outrows if r["group"] == group and r["partition"] == part]
            stats = {key: score([r[key] for r in subset]) for key in ("old_rank", "new_rank", "bm25_rank")}
            for cutoff in (1, 5, 20):
                def hit(row, key):
                    return row[key] is not None and row[key] <= cutoff
                stats[f"paired_hit{cutoff}"] = {
                    "improved": [r["case_id"] for r in subset if hit(r, "new_rank") and not hit(r, "old_rank")],
                    "regressed": [r["case_id"] for r in subset if hit(r, "old_rank") and not hit(r, "new_rank")]}
            metrics[group + "_" + part] = stats
    times = sorted(r["fusion_seconds"] for r in outrows)
    summary = {"plan_sha256": sha(args.plan), "results_sha256": sha(output), "metrics": metrics,
               "fusion_seconds_median": times[len(times) // 2], "fusion_seconds_max": max(times),
               "latency_scope": "offline page-rank fusion only; excludes encoders and retrieval",
               "strict_blind_test": False}
    with Path(str(output) + ".summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("choose", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--baseline", required=True)
        p.add_argument("--output", required=True)
        if command == "evaluate":
            p.add_argument("--plan", required=True)
    args = parser.parse_args()
    (choose if args.command == "choose" else evaluate)(args)


if __name__ == "__main__":
    main()
