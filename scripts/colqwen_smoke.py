"""Run a real local ColQwen2 page/query smoke test on frozen dev candidates."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--case", default="standard-006")
    p.add_argument("--candidates", type=int, default=2)
    p.add_argument("--max-visual-tokens", type=int, default=256)
    args = p.parse_args()
    if args.candidates < 1 or args.candidates > 10 or args.max_visual_tokens < 32:
        raise ValueError("Invalid smoke test bounds")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite existing smoke run")
    output.mkdir(parents=True)
    report = {"status": "started", "case_id": args.case, "device": args.device,
              "candidate_count": args.candidates, "max_visual_tokens": args.max_visual_tokens,
              "model_revision": "2b6ac8fb37f46a49e4841e599583d00ae8a20117",
              "base_revision": "9fe8a713422a7cb4ef79ca77a09b381ee2243101",
              "model_requests": 0, "query_prefix": "Query: "}
    started = time.perf_counter()
    try:
        base, adapter = Path(args.base), Path(args.adapter)
        required = {"model-00001-of-00002.safetensors": (4982048792, "5afb3798d74668030808362aa4a4870cf71a0cfd1780287c9c55e40fb24e51d4"),
                    "model-00002-of-00002.safetensors": (3854759672, "3da90e07b7b0c19944ee2cdcf537fa62ba5b01a9170b193d6b68353e9b033ec3")}
        missing = [name for name, (size, _) in required.items() if not (base / name).is_file() or (base / name).stat().st_size != size]
        if missing:
            raise FileNotFoundError("Missing or incomplete base weights: " + ", ".join(missing))
        if any(sha(base / name) != expected for name, (_, expected) in required.items()):
            raise ValueError("Base weight checksum mismatch")
        adapter_sha = "fc856312174dc99a4c7f88a2c54d9590a3b3f5b5a86e2728d7138c7f4758c6d5"
        if sha(adapter / "adapter_model.safetensors") != adapter_sha:
            raise ValueError("Adapter weight checksum mismatch")
        report["weight_sha256"] = {**{name: digest for name, (_, digest) in required.items()}, "adapter_model.safetensors": adapter_sha}
        from PIL import Image
        import numpy as np
        import psutil
        import torch
        from peft import PeftModel, get_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        from safetensors import safe_open

        candidates = [r for r in rows("runs/irpapers-chroma-baseline-v2-20261106/text-results.jsonl") if r["case_id"] == args.case]
        if len(candidates) != 1 or candidates[0]["partition"] != "dev" or candidates[0]["hybrid"]["status"] != "ok":
            raise ValueError("Expected one successful frozen development case")
        case = candidates[0]
        page_ids = case["hybrid"]["pages"][:args.candidates]
        prepared = Path("data/raw/irpapers-prepared-20261106")
        lookup = {r["page_id"]: r for r in rows(prepared / "pages.jsonl")}
        processor = ColQwen2Processor.from_pretrained(str(adapter), local_files_only=True,
                                                      max_num_visual_tokens=args.max_visual_tokens)
        if processor.query_prefix != "Query: ":
            raise ValueError("Unexpected query prefix")
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model, loading = ColQwen2.from_pretrained(str(base), torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
                                                  device_map=args.device, local_files_only=True, low_cpu_mem_usage=True,
                                                  output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError('Base model loaded with missing, mismatched, or erroneous weights')
        key_mapping = dict(getattr(model, '_checkpoint_conversion_mapping', {}))
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True).eval()
        actual_adapter = get_peft_model_state_dict(model)
        # PEFT applies the base model's official architecture key conversion
        # when loading an older adapter. Compare against that same conversion,
        # rather than requiring the old on-disk names to survive unchanged.
        expected_adapter = load_peft_weights(str(adapter), device='cpu', key_mapping=key_mapping, local_files_only=True)
        with safe_open(str(adapter / 'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
            if len(saved.keys()) != len(expected_adapter):
                raise ValueError('Adapter key conversion lost or collided tensor names')
        if set(actual_adapter) != set(expected_adapter):
            report['adapter_expected_only'] = sorted(set(expected_adapter) - set(actual_adapter))[:12]
            report['adapter_actual_only'] = sorted(set(actual_adapter) - set(expected_adapter))[:12]
            report['adapter_expected_count'] = len(expected_adapter)
            report['adapter_actual_count'] = len(actual_adapter)
            raise ValueError('Loaded adapter tensor names differ from the pinned checkpoint')
        max_adapter_difference = max(float((actual_adapter[k].detach().cpu().float() - expected_adapter[k].float()).abs().max())
                                     for k in expected_adapter)
        if max_adapter_difference > 1e-3:
            raise ValueError('Loaded adapter tensors differ from the pinned checkpoint')
        report.update({'base_unexpected_keys': loading.get('unexpected_keys', []),
                       'base_missing_key_count': len(loading.get('missing_keys', [])),
                       'adapter_key_mapping': key_mapping, 'adapter_tensor_count': len(expected_adapter),
                       'adapter_max_abs_difference': max_adapter_difference})
        del actual_adapter, expected_adapter
        qbatch = processor.process_queries([case["question"]]).to(args.device)
        with torch.inference_mode():
            qout = model(**qbatch)[0].float().cpu().numpy()
        query = qout[qbatch["attention_mask"][0].cpu().numpy().astype(bool)]
        doc_vectors = []
        page_records = []
        for page_id in page_ids:
            page = lookup[page_id]
            path = prepared / page["image_path"]
            if sha(path) != page["image_sha256"]:
                raise ValueError("Candidate page image checksum mismatch")
            with Image.open(path) as img:
                image = img.convert("RGB")
            batch = processor.process_images([image]).to(args.device)
            image.close()
            with torch.inference_mode():
                out = model(**batch)[0].float().cpu().numpy()
            vectors = out[batch["attention_mask"][0].cpu().numpy().astype(bool)]
            if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-3):
                raise ValueError("Invalid ColQwen2 page vectors")
            doc_vectors.append(vectors)
            page_records.append({"page_id": page_id, "image_sha256": page["image_sha256"], "tokens": len(vectors),
                                 "score": float(np.max(query @ vectors.T, axis=1).sum(dtype=np.float64))})
        if not np.isfinite(query).all() or not np.allclose(np.linalg.norm(query, axis=1), 1, atol=2e-3):
            raise ValueError("Invalid ColQwen2 query vectors")
        # Score one page at a time: the reference pads batches with zeros, and
        # padded zeros must not outscore negative similarities of a shorter page.
        reference = [float(processor.score_multi_vector([torch.from_numpy(query)], [torch.from_numpy(d)], device="cpu")[0, 0])
                     for d in doc_vectors]
        if not np.allclose(reference, [p["score"] for p in page_records], atol=1e-3):
            raise ValueError("MaxSim differs from official processor score")
        np.savez_compressed(output / "vectors.npz", query=query, **{f"page_{i}": d for i, d in enumerate(doc_vectors)})
        report.update({"status": "ok", "query_tokens": len(query), "dimension": query.shape[1], "pages": page_records,
                       "maxsim_checked_against_processor": True,
                       "vectors_sha256": sha(output / "vectors.npz"),
                       "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                       "rss_bytes_final": psutil.Process().memory_info().rss,
                       "torch": torch.__version__, "scope": "text-hybrid development candidates only"})
    except Exception as exc:
        report.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
    report["elapsed_seconds"] = time.perf_counter() - started
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
