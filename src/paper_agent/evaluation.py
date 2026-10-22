"""Offline retrieval metrics using gold paragraph units, not overlapping chunks.

Recall follows BEIR's per-query macro averaging. Unlike ignoring missing runs,
all gold queries remain in the denominator. No LLM judge is invoked.
"""
import time
import math
from .storage import digest, read_jsonl, write_json, write_jsonl


def evidence_units(hits):
    return {f"{hit['paper_id']}::{span['paragraph_id']}" for hit in hits for span in hit["spans"]}


def retrieval_metrics(queries, predictions, k=5):
    if not queries:
        raise ValueError("Empty evaluation set")
    ids = [q["query_id"] for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query IDs")
    pred = {p["query_id"]: p for p in predictions}
    if len(pred) != len(predictions) or set(pred) - set(ids):
        raise ValueError("Duplicate or unknown prediction IDs")
    details = []
    for query in queries:
        alternatives = query.get("gold_alternatives") or [query["gold_units"]]
        if not alternatives or any(not a for a in alternatives):
            raise ValueError("Retrieval gold must contain nonempty evidence sets")
        hits = pred.get(query["query_id"], {}).get("hits", [])[:k]
        retrieved = evidence_units(hits)
        # Multiple annotators may give alternative valid evidence paths.
        recall = max(len(set(gold) & retrieved) / len(set(gold)) for gold in alternatives)
        gold_units = set().union(*map(set, alternatives))
        relevances = [int(bool(evidence_units([hit]) & gold_units)) for hit in hits]
        reciprocal = next((1/(i+1) for i, rel in enumerate(relevances) if rel), 0)
        relevant_count = query.get("relevant_chunk_count")
        ideal = sum(1/math.log2(i+2) for i in range(min(k, relevant_count))) if relevant_count is not None else None
        ndcg = sum(rel/math.log2(i+2) for i,rel in enumerate(relevances))/ideal if ideal else None
        details.append({"query_id": query["query_id"], "recall": recall,
                        "mrr": reciprocal, "ndcg": ndcg,
                        "hit": float(recall > 0), "missing": query["query_id"] not in pred,
                        "error": pred.get(query["query_id"], {}).get("error")})
    return {"queries": len(queries), f"Recall@{k}": sum(d["recall"] for d in details)/len(details),
            f"Hit@{k}": sum(d["hit"] for d in details)/len(details),
            f"MRR@{k}": sum(d["mrr"] for d in details)/len(details),
            f"nDCG@{k}": sum(d["ndcg"] for d in details)/len(details) if all(d["ndcg"] is not None for d in details) else None,
            "missing": sum(d["missing"] for d in details),
            "errors": sum(bool(d["error"]) for d in details), "per_query": details}


def evaluate_retriever(retriever, queries_path, output, mode="bm25", top_k=5, candidate_k=30):
    from pathlib import Path
    queries = read_jsonl(queries_path)
    query_hash = digest(queries)
    for query in queries:
        gold = set().union(*map(set, query.get("gold_alternatives") or [query["gold_units"]]))
        query["relevant_chunk_count"] = sum(bool(evidence_units([c]) & gold) for c in retriever.chunks)
    predictions = []
    start = time.perf_counter()
    for query in queries:
        try:
            result = retriever.search(query["query"], mode=mode, top_k=max(10, top_k),
                                      candidate_k=candidate_k, paper_ids=query.get("paper_ids"))
            predictions.append({"query_id": query["query_id"], **result})
        except Exception as exc:
            predictions.append({"query_id": query["query_id"], "hits": [],
                                "error": f"{type(exc).__name__}: {exc}"})
    metrics = retrieval_metrics(queries, predictions, top_k)
    at10 = retrieval_metrics(queries, predictions, 10)
    metrics.update({k:v for k,v in at10.items() if "@10" in k})
    metrics.update(mode=mode, elapsed_seconds=time.perf_counter()-start,
                   corpus_hash=retriever.metadata["corpus_hash"], queries_hash=query_hash,
                   scope="official QASPER text evidence; paragraph coverage proxy; not screenshot reproduction",
                   query_scopes=sorted({q.get("scope", "unspecified") for q in queries}),
                   candidate_k=candidate_k, index_metadata=retriever.metadata)
    write_jsonl(Path(output)/"predictions.jsonl", predictions)
    write_json(Path(output)/"metrics.json", metrics)
    return metrics
