"""Select a bounded page-level CLIP RRF window on development, then evaluate."""
import argparse
import json
import time
from pathlib import Path

from irpapers_optimize import load_dev, rank_gold, ranked, read_frozen, score, sha


def load_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fuse(text, visual, depth, weight):
    return ranked([text, visual[:depth]], [1.0, weight], k=60)


def choose(args):
    root = Path(args.baseline)
    frozen = read_frozen(root)
    text = {r["case_id"]: r for r in load_dev(root / "text-results.jsonl")}
    visual = {r["case_id"]: r for r in load_dev(root / "visual-results.jsonl")}
    if set(text) != set(visual):
        raise ValueError("Development cases differ")
    pairs = [(text[k], visual[k]) for k in sorted(text)]
    if any(a["hybrid"]["status"] != "ok" or b["clip"]["status"] != "ok" for a, b in pairs):
        raise ValueError("A baseline branch failed")
    baselines = {g: score([a["hybrid"]["gold_rank"] for a, _ in pairs if a["group"] == g])
                 for g in ("standard", "visual")}
    candidates = []
    for depth in (5, 20, 50, 100):
        for weight in (0.05, 0.1, 0.25, 0.5, 1.0):
            ranks = {"standard": [], "visual": []}
            for a, b in pairs:
                pages = fuse(a["hybrid"]["pages"], b["clip"]["pages"], depth, weight)
                ranks[a["group"]].append(rank_gold(pages, a))
            stats = {g: score(r) for g, r in ranks.items()}
            stats["all"] = score(ranks["standard"] + ranks["visual"])
            candidates.append({"depth": depth, "weight": weight, "metrics": stats,
                               "eligible": all(stats[g]["hit5"] >= baselines[g]["hit5"] for g in ranks)})
    base_all = score([a["hybrid"]["gold_rank"] for a, _ in pairs])
    eligible = [x for x in candidates if x["eligible"] and x["metrics"]["all"]["mrr"] > base_all["mrr"]]
    winner = max(eligible, key=lambda x: (x["metrics"]["all"]["mrr"], -x["depth"], -x["weight"])) if eligible else None
    plan = {"status": "frozen_before_test_evaluation", "baseline_freeze_sha256": sha(root / "BASELINE-FROZEN.json"),
            "baseline_files": frozen["files"], "script_sha256": sha(__file__), "cases": len(pairs),
            "selection": "development only; best combined MRR; standard and visual Hit@5 must not decrease versus text; nonzero visual weight",
            "strict_blind_test": False, "reason": "historical baseline test aggregates already inspected",
            "baseline_development": {**baselines, "all": base_all}, "candidates": candidates, "winner": winner}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"plan": str(target), "sha256": sha(target), "winner": winner}, ensure_ascii=False))


def evaluate(args):
    root = Path(args.baseline)
    frozen = read_frozen(root)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if (plan["status"] != "frozen_before_test_evaluation" or plan["baseline_freeze_sha256"] != sha(root / "BASELINE-FROZEN.json")
            or plan["baseline_files"] != frozen["files"] or plan["script_sha256"] != sha(__file__)):
        raise ValueError("Frozen selection or source changed")
    if plan["winner"] is None:
        raise ValueError("No development-selected nonzero visual configuration")
    text, visual = load_rows(root / "text-results.jsonl"), load_rows(root / "visual-results.jsonl")
    if len(text) != len(visual):
        raise ValueError("Baseline lengths differ")
    winner = plan["winner"]
    result = []
    start = time.monotonic()
    for a, b in zip(text, visual):
        if a["case_id"] != b["case_id"] or a["hybrid"]["status"] != "ok" or b["clip"]["status"] != "ok":
            raise ValueError("Baseline row mismatch or branch failed")
        pages = fuse(a["hybrid"]["pages"], b["clip"]["pages"], winner["depth"], winner["weight"])
        result.append({"case_id": a["case_id"], "group": a["group"], "partition": a["partition"],
                       "gold_page": a["gold_page"], "gold_status": a["gold_status"],
                       "text_rank": a["hybrid"]["gold_rank"], "equal_fusion_rank": b["fusion"]["gold_rank"],
                       "windowed_rank": rank_gold(pages, a), "top20": pages[:20]})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        for row in result:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = {}
    for group in ("standard", "visual"):
        for partition in ("dev", "test"):
            subset = [r for r in result if r["group"] == group and r["partition"] == partition]
            d = {key: score([r[key] for r in subset]) for key in ("text_rank", "equal_fusion_rank", "windowed_rank")}
            for cutoff in (1, 5, 20):
                hit = lambda r, key: r[key] is not None and r[key] <= cutoff
                d[f"paired_hit{cutoff}_vs_text"] = {
                    "improved": [r["case_id"] for r in subset if hit(r, "windowed_rank") and not hit(r, "text_rank")],
                    "regressed": [r["case_id"] for r in subset if hit(r, "text_rank") and not hit(r, "windowed_rank")]}
            metrics[group + "_" + partition] = d
    summary = {"plan_sha256": sha(args.plan), "results_sha256": sha(target), "metrics": metrics,
               "offline_fusion_seconds": time.monotonic() - start,
               "test_limit": "historical test aggregates were already known; not a strict blind evaluation"}
    with Path(str(target) + ".summary.json").open("x", encoding="utf-8") as stream:
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
