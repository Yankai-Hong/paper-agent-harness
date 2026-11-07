"""Check every stored Chroma page vector against the CLIP encoding checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
from paper_agent.vector_store import ChromaVectorStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root, checkpoint = Path(args.index), Path(args.checkpoint)
    out = Path(args.output)
    if out.exists():
        raise ValueError("Refusing to overwrite verification")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    with (root / "mapping.jsonl").open(encoding="utf-8") as stream:
        mapping = [json.loads(line) for line in stream if line.strip()]
    vectors = np.load(checkpoint / "vectors.npy", mmap_mode="r")
    store = ChromaVectorStore(root / "vectors", metadata["vector_contract"])
    if len(mapping) != len(vectors) or store.collection.count() != len(mapping):
        raise ValueError("Page vector count mismatch")
    difference = 0.0
    for start in range(0, len(mapping), 200):
        ids = [row["page_id"] for row in mapping[start:start + 200]]
        data = store.collection.get(ids=ids, include=["embeddings"])
        lookup = dict(zip(data["ids"], data["embeddings"]))
        if set(ids) != set(lookup):
            raise ValueError("Page IDs differ")
        for i, page_id in enumerate(ids):
            difference = max(difference, float(np.max(np.abs(vectors[start + i] - lookup[page_id]))))
    result = {"count": len(mapping), "dimension": vectors.shape[1], "max_absolute_difference": difference,
              "index": str(root), "checkpoint": str(checkpoint), "model_provenance": metadata["model_provenance"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
