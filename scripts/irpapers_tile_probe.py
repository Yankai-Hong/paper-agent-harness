"""Small frozen development probe of two provenance-preserving CLIP page tiles."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path, dev_only=False):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if dev_only and '"partition": "dev"' not in line:
                continue
            if line.strip():
                yield json.loads(line)


def choose(args):
    root = Path(args.baseline)
    from irpapers_optimize import read_frozen
    frozen = read_frozen(root)
    text = list(rows(root / "text-results.jsonl", True))
    clip = {r["case_id"]: r for r in rows(root / "visual-results.jsonl", True)}
    selected = [r for group in ("standard", "visual") for r in sorted(
        (r for r in text if r["group"] == group), key=lambda r: r["case_id"])[:4]]
    cases = []
    for row in selected:
        visual = clip[row["case_id"]]
        if row["hybrid"]["status"] != "ok" or visual["clip"]["status"] != "ok":
            raise ValueError("A candidate branch failed")
        candidates = list(dict.fromkeys(row["hybrid"]["pages"][:10] + visual["clip"]["pages"][:10]))
        cases.append({"case_id": row["case_id"], "group": row["group"], "candidate_pages": candidates})
    plan = {"status": "frozen_before_tile_encoding", "baseline_files": frozen["files"],
            "baseline_freeze_sha256": sha(root / "BASELINE-FROZEN.json"), "script_sha256": sha(__file__),
            "selection": "first four standard and first four visual development case IDs, no gold filtering",
            "candidate_policy": "deduplicated top 10 text pages plus top 10 CLIP pages from the complete corpus",
            "tile_policy": "top 60% and bottom 60% of page, each padded white to square before pinned CLIP processor",
            "visual_weight": 0.25, "rrf_k": 60, "cases": cases}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"plan": str(out), "sha256": sha(out), "cases": [c["case_id"] for c in cases],
                      "unique_candidate_pages": len({p for c in cases for p in c["candidate_pages"]})}))


def run(args):
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    from irpapers_optimize import read_frozen, ranked
    sys.path.insert(0, str(Path('src').resolve()))
    from paper_agent.visual_retrieval import _model_manifest
    root = Path(args.baseline)
    frozen = read_frozen(root)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if (plan["status"] != "frozen_before_tile_encoding" or plan["baseline_files"] != frozen["files"]
            or plan["baseline_freeze_sha256"] != sha(root / "BASELINE-FROZEN.json") or plan["script_sha256"] != sha(__file__)):
        raise ValueError("Tile plan mismatch")
    baseline_meta = json.loads((root / 'visual-results.jsonl.summary.json').read_text(encoding='utf-8'))['visual_metadata']
    provenance = {'image': _model_manifest('models/multimodal-v1/clip-image-v1'),
                  'text': _model_manifest('models/multimodal-v1/clip-text-v1')}
    if provenance != baseline_meta['model_provenance']:
        raise ValueError('CLIP snapshots differ from frozen baseline')
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite existing tile output")
    output.mkdir(parents=True)
    text = {r["case_id"]: r for r in rows(root / "text-results.jsonl", True)}
    visual = {r["case_id"]: r for r in rows(root / "visual-results.jsonl", True)}
    prepared = Path("data/raw/irpapers-prepared-20261106")
    pages = {r["page_id"]: r for r in rows(prepared / "pages.jsonl")}
    names = list(dict.fromkeys(page for case in plan["cases"] for page in case["candidate_pages"]))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = SentenceTransformer("models/multimodal-v1/clip-image-v1", device=args.device, local_files_only=True)
    vectors = []
    mapping = []
    for j in range(0, len(names), 2):
        images = []
        for page_id in names[j:j + 2]:
            page = pages[page_id]
            path = prepared / page["image_path"]
            if sha(path) != page["image_sha256"]:
                raise ValueError("Candidate image hash mismatch")
            with Image.open(path) as image:
                im = image.convert("RGB")
            width, height = im.size
            for number, box in enumerate(((0, 0, width, round(height * .6)), (0, round(height * .4), width, height))):
                crop = im.crop(box)
                side = max(crop.size)
                square = Image.new("RGB", (side, side), "white")
                square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
                crop.close()
                images.append(square)
                mapping.append({"tile_id": page_id + ":" + str(number), "page_id": page_id, "paper_id": page["paper_id"],
                                "page_number": page["page_number"], "image_sha256": page["image_sha256"],
                                "box_pixels": list(box), "source_width": width, "source_height": height,
                                "padding": "white to centered square"})
            im.close()
        batch = model.encode(images, batch_size=4, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        vectors.extend(np.asarray(batch, dtype=np.float32))
        for image in images:
            image.close()
        if (j + 2) % 20 == 0:
            print("pages encoded", min(j + 2, len(names)), "of", len(names), flush=True)
    del model
    torch.cuda.empty_cache()
    vectors = np.asarray(vectors, dtype=np.float32)
    np.save(output / "vectors.npy", vectors, allow_pickle=False)
    with (output / "mapping.jsonl").open("x", encoding="utf-8") as stream:
        for row in mapping:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    model = SentenceTransformer("models/multimodal-v1/clip-text-v1", device=args.device, local_files_only=True)
    queries = model.encode([text[c["case_id"]]["question"] for c in plan["cases"]], batch_size=8,
                           normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    lookup = {name: i for i, name in enumerate(names)}
    results = []
    for c, q in zip(plan["cases"], queries):
        row, old = text[c["case_id"]], visual[c["case_id"]]
        scores = vectors @ q
        tile_rank = sorted(c["candidate_pages"], key=lambda p: (-float(max(scores[lookup[p] * 2:lookup[p] * 2 + 2])), p))
        text_rank = [p for p in row["hybrid"]["pages"] if p in set(c["candidate_pages"])]
        fused = ranked([text_rank, tile_rank], [1.0, plan["visual_weight"]], k=plan["rrf_k"])
        def rank(pages):
            return pages.index(row["gold_page"]) + 1 if row["gold_status"] == "consistent" and row["gold_page"] in pages else None
        results.append({"case_id": row["case_id"], "group": row["group"], "gold_status": row["gold_status"],
                        "candidate_hit": rank(tile_rank) is not None, "candidate_count": len(tile_rank),
                        "text_global_gold_rank": row["hybrid"]["gold_rank"], "clip_global_gold_rank": old["clip"]["gold_rank"],
                        "tile_gold_rank_in_candidates": rank(tile_rank), "fused_gold_rank_in_candidates": rank(fused),
                        "tile_top5": tile_rank[:5], "fused_top5": fused[:5]})
    report = {"plan_sha256": sha(args.plan), "vectors_sha256": sha(output / "vectors.npy"),
              "mapping_sha256": sha(output / "mapping.jsonl"), "candidate_pages": len(names), "tiles": len(mapping),
              "model_provenance": provenance,
              "dimension": vectors.shape[1], "device": args.device, "elapsed_seconds": time.perf_counter() - start,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "rss_bytes_final": psutil.Process().memory_info().rss,
              "scope": "eight development cases, candidate-only diagnostic; not full-corpus tile retrieval", "results": results}
    with (output / "results.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ("choose", "run"):
        s = sub.add_parser(cmd)
        s.add_argument("--baseline", required=True)
        s.add_argument("--output", required=True)
        if cmd == "run":
            s.add_argument("--plan", required=True)
            s.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    (choose if a.command == "choose" else run)(a)


if __name__ == "__main__":
    main()
