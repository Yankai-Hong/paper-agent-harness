"""Isolated CLIP operations; one model per process, no model vectors in logs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("model_id", choices=["sentence-transformers/clip-ViT-B-32", "sentence-transformers/clip-ViT-B-32-multilingual-v1"])
    download.add_argument("revision", help="immutable Hugging Face commit SHA")
    download.add_argument("directory")
    encode = sub.add_parser("encode")
    encode.add_argument("request")
    split = sub.add_parser("split")
    split.add_argument("request")
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download
        import re
        if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            raise ValueError("revision must be an immutable 40-character commit")
        target = Path(args.directory)
        provenance = {"model_id": args.model_id, "revision": args.revision}
        marker = target / "provenance.json"
        if target.exists() and any(target.iterdir()):
            if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != provenance:
                raise ValueError("Refusing to overwrite a different model directory")
        target.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(json.dumps(provenance), encoding="utf-8")
        # The safetensors weights are sufficient; do not download duplicate
        # PyTorch, TensorFlow, ONNX or OpenVINO exports.
        snapshot_download(repo_id=args.model_id, revision=args.revision, local_dir=str(target),
                          ignore_patterns=["*.bin", "*.onnx", "*.h5", "openvino/*", ".gitattributes"])
        (target / "snapshot.complete.json").write_text(json.dumps(provenance), encoding="utf-8")
        return
    if args.command == "split":
        from transformers import AutoTokenizer
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(request["model"], local_files_only=True, use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError("Fast tokenizer with character offsets required")
        # Sentence Transformers' saved maximum, not tokenizer's often unbounded default.
        config = json.loads((Path(request["model"]) / "0_Transformer" / "sentence_bert_config.json").read_text(encoding="utf-8")) if (Path(request["model"]) / "0_Transformer" / "sentence_bert_config.json").exists() else json.loads((Path(request["model"]) / "sentence_bert_config.json").read_text(encoding="utf-8"))
        limit = int(config["max_seq_length"])
        window = limit - tokenizer.num_special_tokens_to_add(pair=False)
        overlap = min(16, window // 4)
        if window < 2:
            raise ValueError("Invalid tokenizer limit")
        rows = []
        for text in request["inputs"]:
            offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)["offset_mapping"]
            offsets = [(a, b) for a, b in offsets if b > a]
            spans = []
            for i in range(0, len(offsets), window - overlap):
                batch = offsets[i:i + window]
                if not batch:
                    break
                spans.append([batch[0][0], batch[-1][1]])
                if i + window >= len(offsets):
                    break
            rows.append(spans)
        Path(request["output"]).write_text(json.dumps({"spans": rows, "limit": limit, "window": window, "overlap": overlap}), encoding="utf-8")
        return
    import numpy as np
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    model = SentenceTransformer(request["model"], device=request["device"], local_files_only=True)
    if request["kind"] == "images":
        images = []
        for path in request["inputs"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        items = images
    elif request["kind"] == "texts":
        items = request["inputs"]
    else:
        raise ValueError("Unknown input kind")
    vectors = model.encode(items, batch_size=8, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    np.save(request["output"], np.asarray(vectors, dtype=np.float32), allow_pickle=False)


if __name__ == "__main__":
    main()
