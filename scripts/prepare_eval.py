"""Build the unified data/eval/ dataset directory with explicit label provenance.

No model calls. The retrieval set reuses QASPER official annotations (usable gold).
Intent / rewrite / boundary / memory labels are derived from the source-linked draft
dialogues and are marked draft_unreviewed: they are protocol-complete but MUST be
human-reviewed before any effectiveness claim (see data/eval/README.md).
"""
import argparse
import shutil
from pathlib import Path

from paper_agent.storage import read_jsonl, write_jsonl, write_json, digest


def session_history(dialogues, row):
    return [{"role": "user", "content": d["question"], "turn_id": d["turn_id"],
             "session_id": d["session_id"], "turn": d["turn"]}
            for d in sorted(dialogues, key=lambda d: d["turn"])
            if d["session_id"] == row["session_id"] and d["turn"] < row["turn"]]


def prepare(source_qasper="data/qasper", source_dialogues="data/dialogues-draft", output="data/eval/v0.2"):
    out = Path(output)
    if (out / "manifest.json").exists():
        raise FileExistsError("Version already exists; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)

    # 1. Retrieval: official QASPER annotations (usable as gold for retrieval metrics).
    retrieval = read_jsonl(Path(source_qasper) / "retrieval.jsonl")
    write_jsonl(out / "retrieval.jsonl", retrieval)

    dialogues = read_jsonl(Path(source_dialogues) / "dialogues.jsonl")

    # 2. Dialogs: full 30x4 source-linked drafts.
    write_jsonl(out / "dialogs.jsonl", dialogues)

    # 3. Intent: first 100 turns, provisional intent labels (draft).
    intent = [{"id": r["turn_id"], "question": r["question"],
               "session_id": r["session_id"], "turn": r["turn"], "history": session_history(dialogues, r),
               "intent": r["provisional_intent"],
               "review_status": "draft_unreviewed",
               "label_source": "template-derived provisional intent; requires human review"}
              for r in dialogues[:100]]
    write_jsonl(out / "intent.jsonl", intent)

    # 4. Query rewrite: follow-up turns (turn>=2) carry coreference/ellipsis.
    rewrite = [{"id": r["turn_id"], "question": r["question"],
                "session_id": r["session_id"], "turn": r["turn"],
                "history": session_history(dialogues, r),
                "required_entities": r["source_paper_ids"],
                "expected_query": None,
                "review_status": "draft_unreviewed",
                "label_source": "expected standalone query requires human review"}
               for i, r in enumerate(dialogues) if r["turn"] >= 2]
    write_jsonl(out / "query_rewrite.jsonl", rewrite)

    # 5. Boundary: 4th turns + first 25 openings, provisional decisions (draft).
    boundary = [{"id": r["turn_id"], "question": r["question"],
                 "session_id": r["session_id"], "turn": r["turn"], "history": session_history(dialogues, r),
                 "decision": r["provisional_decision"] or "answer",
                 "review_status": "draft_unreviewed",
                 "label_source": "template-derived provisional decision; requires human review"}
                for r in dialogues if r["turn"] == 4]
    boundary += [{"id": r["turn_id"], "question": r["question"],
                  "session_id": r["session_id"], "turn": r["turn"], "history": [],
                  "decision": "answer",
                  "review_status": "draft_unreviewed",
                  "label_source": "answerable opening; requires human review"}
                 for r in dialogues if r["turn"] == 1][:25]
    write_jsonl(out / "boundary.jsonl", boundary)

    # 6. Memory: checkpoint schema over the first eight sessions (draft checkpoints).
    memory = []
    for sid in sorted({r["session_id"] for r in dialogues})[:8]:
        opening = next(r for r in dialogues if r["session_id"] == sid and r["turn"] == 1)
        for turn in range(1, 11):
            memory.append({"id": f"mem-{sid}-t{turn}", "session_id": sid,
                       "question": opening["question"] if turn in {1,10} else
                           "请继续核对刚才那篇论文的原文证据，指出该问题可以回答和不能回答的部分，并保留来源引用。",
                       "turn": turn,
                       "checkpoints": {"discussed_papers": opening["source_paper_ids"],
                                       "fact_source_papers": opening["source_paper_ids"]},
                       "expected": {"resolved_paper_id": opening["source_paper_ids"][0]},
                       "review_status": "draft_unreviewed",
                       "label_source": "checkpoints require running the pipeline and human review"})
    write_jsonl(out / "memory.jsonl", memory)

    manifest = {
        "version": out.name + "-draft", "generated_from": {"qasper": source_qasper, "dialogues": source_dialogues},
        "counts": {"retrieval": len(retrieval), "dialogs": len(dialogues),
                   "intent": len(intent), "query_rewrite": len(rewrite),
                   "boundary": len(boundary), "memory": len(memory)},
        "retrieval_label_source": "QASPER v0.3 official annotations (CC BY 4.0)",
        "other_label_status": "draft_unreviewed; provisional labels only, not ground truth",
        "hashes": {name: digest(read_jsonl(out / name)) for name in
                   ["retrieval.jsonl", "dialogs.jsonl", "intent.jsonl",
                    "query_rewrite.jsonl", "boundary.jsonl", "memory.jsonl"]},
    }
    write_json(out / "manifest.json", manifest)
    print("Prepared data/eval/:", manifest["counts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasper", default="data/qasper")
    parser.add_argument("--dialogues", default="data/dialogues-draft")
    parser.add_argument("--output", default="data/eval/v0.2")
    args = parser.parse_args()
    prepare(args.qasper, args.dialogues, args.output)
