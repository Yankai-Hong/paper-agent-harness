"""Re-encode frozen development examples and audit the page CLIP snapshot."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite existing diagnosis")
    from PIL import Image
    import torch
    from sentence_transformers import SentenceTransformer

    root = Path("data/raw/irpapers-prepared-20261106")
    checkpoint = Path("data/raw/irpapers-page-clip-checkpoint-20261106")
    pages = rows(root / "pages.jsonl")
    mapping = rows(checkpoint / "mapping.jsonl")
    indexed = rows("indexes/irpapers-chroma-page-clip-baseline-20261106/mapping.jsonl")
    fields = ("page_id", "paper_id", "pdf_id", "page_number", "image_path", "image_sha256", "width", "height")
    expected = [{k: p[k] for k in fields} for p in pages]
    actual = [{k: p[k] for k in fields} for p in mapping]
    if expected != actual or mapping != indexed:
        raise ValueError("Prepared, checkpoint, and Chroma mappings differ")
    vectors = np.load(checkpoint / "vectors.npy", mmap_mode="r")
    if vectors.shape != (len(mapping), 512):
        raise ValueError("Unexpected vector shape")
    records = rows("runs/irpapers-chroma-baseline-v2-20261106/visual-results.jsonl")
    chosen = [r for r in records if r["case_id"] in {"standard-006", "standard-007", "visual-008", "visual-019"}]
    if len(chosen) != 4 or any(r["partition"] != "dev" for r in chosen):
        raise ValueError("Expected four frozen development cases")
    positions = {p["page_id"]: n for n, p in enumerate(mapping)}
    names = list(dict.fromkeys(p for r in chosen for p in (r["gold_page"], r["clip"]["pages"][0])))
    images = []
    for name in names:
        page = mapping[positions[name]]
        path = root / page["image_path"]
        if digest(path) != page["image_sha256"]:
            raise ValueError("Image digest mismatch")
        with Image.open(path) as im:
            im.load()
            images.append(im.convert("RGB"))
    start = time.monotonic()
    model = SentenceTransformer("models/multimodal-v1/clip-image-v1", device=args.device, local_files_only=True)
    image_vectors = model.encode(images, batch_size=2, normalize_embeddings=True, convert_to_numpy=True)
    del model
    for im in images:
        im.close()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    text_model = SentenceTransformer("models/multimodal-v1/clip-text-v1", device=args.device, local_files_only=True)
    query_vectors = text_model.encode([r["question"] for r in chosen], batch_size=4, normalize_embeddings=True, convert_to_numpy=True)
    result = []
    for row, q in zip(chosen, query_vectors):
        scores = vectors @ q
        ranking = np.argsort(-scores, kind="stable")
        ordered = [mapping[int(i)]["page_id"] for i in ranking]
        result.append({"case_id": row["case_id"], "question": row["question"], "gold_page": row["gold_page"],
                       "baseline_gold_rank": row["clip"]["gold_rank"], "recomputed_gold_rank": ordered.index(row["gold_page"]) + 1,
                       "baseline_top5": row["clip"]["pages"][:5], "recomputed_top5": ordered[:5],
                       "target_score": float(scores[positions[row["gold_page"]]]), "top_score": float(scores[ranking[0]])})
    differences = [{"page_id": name, "max_abs_difference": float(np.max(np.abs(image_vectors[i] - vectors[positions[name]]))),
                    "cosine": float(image_vectors[i] @ vectors[positions[name]])} for i, name in enumerate(names)]
    report = {"mapping_count": len(mapping), "mapping_identical": True, "sample_image_hashes_verified": len(names),
              "sample_reencoded_images": differences, "examples": result, "device": args.device,
              "seconds": time.monotonic() - start, "image_processor": "RGB, resize shortest edge 224, center crop 224x224",
              "image_model_sha256": "85d6c4d8881ccf203ce6b9c9f664fca849ec175e84ec939592baa35fb9c418e3",
              "text_model_sha256": "3739808321eb3cedc449e4c732dea9088c55c83762b6f0888077be761a603d20"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
