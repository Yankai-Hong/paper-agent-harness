"""Fetch the official QASPER archive and derive a deterministic retrieval subset.

No training, generated labels, paid API, or automatic manual-review claims.
Archive URL is from allenai/qasper's official dataset loader (v0.3).
Reads a named member without extracting arbitrary archive paths.
"""
import argparse
import hashlib
import json
import random
import tarfile
import urllib.request
from pathlib import Path

from paper_agent.storage import write_json, write_jsonl

URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"


def normalize(text):
    return " ".join(text.split())


def prepare(archive, output, paper_count=100, query_count=100, seed=42):
    with tarfile.open(archive, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if Path(m.name).name == "qasper-test-v0.3.json")
        with tar.extractfile(member) as stream:
            data = json.load(stream)
    keys = sorted(data)
    random.Random(seed).shuffle(keys)
    papers, queries, excluded = [], [], {"unanswerable_or_no_text_evidence": 0, "unmapped_evidence": 0}
    for key in keys[:paper_count]:
        source = data[key]
        paragraphs = []
        if source.get("abstract"):
            paragraphs.append({"paragraph_id": "abstract", "section": "Abstract", "text": source["abstract"]})
        for section_index, section in enumerate(source["full_text"]):
            for index, text in enumerate(section["paragraphs"]):
                paragraphs.append({"paragraph_id": f"s{section_index}-p{index}",
                                   "section": section["section_name"], "text": text})
        papers.append({"paper_id": key, "title": source["title"], "paragraphs": paragraphs,
                       "source_url": f"https://arxiv.org/abs/{key}", "dataset": "QASPER v0.3 test",
                       "dataset_license": "CC BY 4.0", "review_status": "official annotations; no local human review"})
        lookup = {}
        for paragraph in paragraphs:
            lookup.setdefault(normalize(paragraph["text"]), []).append(f"{key}::{paragraph['paragraph_id']}")
        for qa in source["qas"]:
            alternatives = []
            for annotation in qa["answers"]:
                answer = annotation["answer"]
                if answer["unanswerable"] or not answer.get("evidence"):
                    continue
                gold = []
                valid = True
                for evidence in answer["evidence"]:
                    matched = lookup.get(normalize(evidence), [])
                    if len(matched) != 1:
                        valid = False
                        break
                    gold.extend(matched)
                if valid and gold:
                    alternatives.append(sorted(set(gold)))
                elif not valid:
                    excluded["unmapped_evidence"] += 1
            if alternatives:
                queries.append({"query_id": qa["question_id"], "query": qa["question"],
                                "gold_units": alternatives[0], "gold_alternatives": alternatives,
                                "source_paper_id": key, "paper_ids": [key], "scope": "known_paper",
                                "annotation_source": "QASPER official; answer text not indexed"})
            else:
                excluded["unanswerable_or_no_text_evidence"] += 1
    random.Random(seed).shuffle(queries)
    queries = queries[:query_count]
    if len(papers) != paper_count or len(queries) != query_count:
        raise ValueError(f"Insufficient eligible samples: {len(papers)} papers, {len(queries)} queries")
    output = Path(output)
    write_jsonl(output/"papers.jsonl", papers)
    write_jsonl(output/"retrieval.jsonl", queries)
    write_jsonl(output/"retrieval-global.jsonl", [
        {**{k: v for k, v in q.items() if k != "paper_ids"}, "scope": "global_corpus_stress"} for q in queries])
    write_json(output/"manifest.json", {"source": URL, "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
               "dataset_license": "CC BY 4.0", "citation": "Dasigi et al., NAACL 2021, QASPER",
               "seed": seed, "papers": len(papers), "queries": len(queries), "excluded": excluded,
               "scope": "English single-turn known-paper retrieval; global corpus stress view saved separately; not the custom multi-turn test set"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/raw/qasper-test-v0.3.tgz")
    parser.add_argument("--output", default="data/qasper")
    parser.add_argument("--papers", type=int, default=100)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    archive = Path(args.archive)
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=90) as response, temporary.open("wb") as stream:
            import shutil
            shutil.copyfileobj(response, stream)
        temporary.replace(archive)
    prepare(archive, args.output, args.papers, args.queries)
    print(json.dumps({"papers": args.papers, "queries": args.queries, "output": args.output}))
