"""One offline GPU run: index + four retrieval baselines, with frozen inputs.

No API usage. All modes use identical chunks, questions, and candidate budget.
"""
import argparse
import gc
import json
import hashlib
import platform
import time
from pathlib import Path

from paper_agent.corpus import Chunker, ingest_jsonl
from paper_agent.retrieval import BGEEncoder, BGEReranker, Retriever, build_index
from paper_agent.evaluation import evaluate_retriever
from paper_agent.storage import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/qasper")
    parser.add_argument("--output", default="reports/reproduced-retrieval")
    parser.add_argument("--index", default="indexes/bge")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder", default="models/bge-m3")
    parser.add_argument("--reranker", default="models/bge-reranker-v2-m3")
    parser.add_argument("--reuse-index", action="store_true")
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but not available")
    torch.set_num_threads(4)
    start = time.perf_counter()
    data, output = Path(args.data), Path(args.output)
    root = Path(__file__).resolve().parents[1]
    write_json(output/"run_config.json", {"arguments": vars(args),
        "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in (root/"src", root/"scripts") for p in folder.rglob("*.py")},
        "encoder_provenance": json.loads((Path(args.encoder)/"provenance.json").read_text()),
        "dataset_manifest": json.loads((data/"manifest.json").read_text()),
        "evaluation_scope": "known-paper QASPER; fixed defaults, no test-set parameter tuning"})
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True, local_files_only=True)
    if not args.reuse_index:
        ingest_jsonl(data/"papers.jsonl", data/"chunks.jsonl", Chunker(tokenizer=tokenizer, tokenizer_id=args.encoder))
    encoder = BGEEncoder(args.encoder, args.device, args.batch_size)
    if not args.reuse_index:
        print("Building exact FAISS index", flush=True)
        build_index(data/"chunks.jsonl", args.index, encoder)
    retriever = Retriever(args.index, encoder)
    summary = {}
    for mode in ["bm25", "dense", "hybrid", "rerank"]:
        if mode == "rerank":
            retriever.reranker = BGEReranker(args.reranker, args.device, args.batch_size)
        print(f"Evaluating {mode}", flush=True)
        metrics = evaluate_retriever(retriever, data/"retrieval.jsonl", output/mode, mode)
        summary[mode] = {key: metrics[key] for key in ["queries", "Recall@5", "Hit@5", "errors", "elapsed_seconds"]}
        print(json.dumps(summary[mode]), flush=True)
        write_json(output/"summary.json", summary)
    write_json(output/"environment.json", {"python": platform.python_version(), "torch": torch.__version__,
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "elapsed_seconds": time.perf_counter()-start, "batch_size": args.batch_size,
        "max_length": 768, "paid_api_calls": 0})
    write_json(output/"reranker_provenance.json", json.loads((Path(args.reranker)/"provenance.json").read_text()))
