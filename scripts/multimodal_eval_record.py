"""Create/score a local multimodal evaluation record without calling any model.

The predictions input is a JSON object with a ``cases`` mapping by case id.
Each result can contain ``retrieval_complete``, ``top5``, ``answer``,
``citations``, ``citation_location``, ``semantic_support``, ``usage``,
``elapsed_seconds`` and ``failure``. The last two judgments are deliberately
not inferred from citation syntax or retrieval. No files are sent anywhere.
"""
import argparse
import json
import re
from pathlib import Path


def figure_key(value):
    match = re.search(r"\b(fig(?:ure)?|table)\.?\s*(\d+)\b", str(value or ""), re.I)
    if not match:
        return None
    return ("figure" if match.group(1).lower().startswith("fig") else "table", int(match.group(2)))


def score_visual(target, top5, complete):
    wanted = figure_key(target["locator"])
    if wanted is None:
        return None  # Page alone is not sufficient to prove text coverage.
    ambiguous = False
    for evidence in (top5 or [])[:5]:
        if evidence.get("paper_id") != target["paper_id"] or evidence.get("page") != target["page"]:
            continue
        got = figure_key(evidence.get("figure_number"))
        if got == wanted:
            return True
        if got is None:
            ambiguous = True
    return None if ambiguous or not complete else False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="data/eval/multimodal-v1/cases.json")
    parser.add_argument("--predictions", help="Optional local results JSON; omitted creates an unexecuted template")
    parser.add_argument("--output", required=True, help="New output path; existing results are never overwritten")
    args = parser.parse_args()
    source = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8")) if args.predictions else {}
    rows = []
    for case in source["cases"] + source["image_queries"]:
        row = predictions.get("cases", {}).get(case["id"], {})
        targets = case.get("evidence") or ([{"page": case["target_page"], "locator": case["target_locator"]}] if "target_page" in case else [])
        checks = [{"locator": t["locator"], "page": t["page"], "hit_at_5": score_visual({**t, "paper_id": case["paper_id"]}, row.get("top5", []), row.get("retrieval_complete", False))} for t in targets]
        rows.append({"id": case["id"], "paper_id": case["paper_id"], "executed": bool(row),
                     "top5": row.get("top5"), "target_checks": checks,
                     "answer": row.get("answer"), "answer_correct": row.get("answer_correct"), "citations": row.get("citations"),
                     "citation_location": row.get("citation_location"), "semantic_support": row.get("semantic_support"),
                     "usage": row.get("usage"), "calls": row.get("calls"), "elapsed_seconds": row.get("elapsed_seconds"), "failure": row.get("failure"),
                     "review_note": row.get("review_note")})
    output = {"case_set": source["version"], "note": "null means unknown, not success; location and semantic support require independent review", "cases": rows}
    with Path(args.output).open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
