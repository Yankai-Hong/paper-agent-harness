"""Record the runtime dependency closure, not unrelated Conda packages."""
import importlib.metadata as metadata
from pathlib import Path
from packaging.requirements import Requirement


roots = ["numpy", "rank-bm25", "faiss-cpu", "pypdf", "jsonschema", "FlagEmbedding",
         "transformers", "pytest", "torch", "hf-xet", "python-dotenv"]
seen = {}
pending = list(roots)
while pending:
    name = pending.pop()
    key = name.lower().replace("_", "-")
    if key in seen:
        continue
    distribution = metadata.distribution(name)
    seen[key] = f"{distribution.metadata['Name']}=={distribution.version}"
    for dependency in distribution.requires or []:
        req = Requirement(dependency)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            pending.append(req.name)
Path("requirements-lock.txt").write_text(
    "# Windows/Python 3.10 runtime closure; install CUDA PyTorch from its official wheel index first.\n"
    + "\n".join(seen[key] for key in sorted(seen)) + "\n", encoding="utf-8")
print(f"Recorded {len(seen)} package versions")
