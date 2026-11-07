"""Freeze and audit an unfiltered text-plus-CLIP page candidate union."""
import argparse
import json
from pathlib import Path

from irpapers_optimize import load_dev, read_frozen, sha


def union(a, b, depth):
    return list(dict.fromkeys(a[:depth] + b[:depth]))


def choose(args):
    root = Path(args.baseline)
    frozen = read_frozen(root)
    text = list(load_dev(root / "text-results.jsonl"))
    visual = {r["case_id"]: r for r in load_dev(root / "visual-results.jsonl")}
    if set(r["case_id"] for r in text) != set(visual):
        raise ValueError("Development cases do not match")
    profiles = {}
    for depth in (20, 50, 100):
        groups = {}
        for group in ("standard", "visual"):
            subset = [r for r in text if r["group"] == group]
            groups[group] = {"n": len(subset), "candidate_hits": 0, "total_candidates": 0}
            for row in subset:
                clip = visual[row["case_id"]]
                if row["hybrid"]["status"] != "ok" or clip["clip"]["status"] != "ok":
                    raise ValueError("Baseline branch failed")
                candidate = union(row["hybrid"]["pages"], clip["clip"]["pages"], depth)
                groups[group]["total_candidates"] += len(candidate)
                groups[group]["candidate_hits"] += int(row["gold_status"] == "consistent" and row["gold_page"] in candidate)
            groups[group]["mean_candidates"] = groups[group]["total_candidates"] / len(subset)
        profiles[str(depth)] = groups
    plan = {"status": "frozen_before_test_evaluation", "baseline_freeze_sha256": sha(root / "BASELINE-FROZEN.json"),
            "baseline_files": frozen["files"], "script_sha256": sha(__file__), "development_profiles": profiles,
            "text_depth": 50, "clip_depth": 50, "max_candidates": 100,
            "selection": "50+50 balances development visual candidate coverage and per-query encoding size; deduplicate by page ID",
            "candidate_source": "full-corpus text hybrid and CLIP rankings; no gold filtering",
            "maxsim_scope": "candidate pages only; no full-corpus claim",
            "strict_blind_test": False, "reason": "historical baseline test aggregates already known"}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"plan": str(out), "sha256": sha(out), "development_profiles": profiles}))


def evaluate(args):
    root = Path(args.baseline)
    frozen = read_frozen(root)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if (plan["status"] != "frozen_before_test_evaluation" or plan["baseline_freeze_sha256"] != sha(root / "BASELINE-FROZEN.json")
            or plan["baseline_files"] != frozen["files"] or plan["script_sha256"] != sha(__file__)
            or plan["text_depth"] != plan["clip_depth"]):
        raise ValueError("Candidate plan mismatch")
    with (root / "text-results.jsonl").open(encoding="utf-8") as stream:
        text = [json.loads(line) for line in stream if line.strip()]
    with (root / "visual-results.jsonl").open(encoding="utf-8") as stream:
        visual = [json.loads(line) for line in stream if line.strip()]
    if len(text) != len(visual):
        raise ValueError("Baseline result count mismatch")
    result = []
    for a, b in zip(text, visual):
        if a["case_id"] != b["case_id"] or a["hybrid"]["status"] != "ok" or b["clip"]["status"] != "ok":
            raise ValueError("Baseline row mismatch or failed branch")
        pages = union(a["hybrid"]["pages"], b["clip"]["pages"], plan["text_depth"])
        result.append({"case_id": a["case_id"], "group": a["group"], "partition": a["partition"],
                       "gold_status": a["gold_status"], "candidate_hit": a["gold_status"] == "consistent" and a["gold_page"] in pages,
                       "candidate_count": len(pages), "pages": pages})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        for row in result:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {}
    for group in ("standard", "visual"):
        for part in ("dev", "test"):
            subset = [r for r in result if r["group"] == group and r["partition"] == part]
            summary[group + "_" + part] = {"n": len(subset), "hits": sum(r["candidate_hit"] for r in subset),
                                           "mean_candidates": sum(r["candidate_count"] for r in subset) / len(subset),
                                           "unique_pages": len({page for r in subset for page in r["pages"]}),
                                           "misses": [r["case_id"] for r in subset if not r["candidate_hit"]]}
    data = {"plan_sha256": sha(args.plan), "results_sha256": sha(out), "candidate_coverage": summary,
            "maxsim_executed": False, "note": "Candidate coverage alone is not a ColQwen2 result."}
    with Path(str(out) + ".summary.json").open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    print(json.dumps(data, ensure_ascii=False))


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
