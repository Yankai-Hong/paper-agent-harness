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
