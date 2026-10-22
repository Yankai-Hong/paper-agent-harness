"""Download official BAAI artifacts to workspace with resolved revision records."""
import argparse
import hashlib
import urllib.request
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from paper_agent.storage import write_json

MODEL_REVISIONS = {
    "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
}

def file_sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def ranged_download(url, destination, size, expected):
    """Bounded ranges avoid stalls on large HTTP streams; verify every response."""
    temporary = destination.with_suffix(destination.suffix + ".ranged-download")
    checkpoint = destination.with_suffix(destination.suffix + ".ranges.json")
    block_size = 4 * 1024 * 1024
    completed = set()
    if temporary.exists() and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("sha256") == expected and saved.get("size") == size:
            completed = set(saved["completed"])
    if not completed:
        with temporary.open("wb") as stream:
            stream.truncate(size)
    lock = threading.Lock()
    total = (size + block_size-1)//block_size
    def fetch(index):
        start = index * block_size
        end = min(size, start + block_size)-1
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Unexpected range response")
                    data = response.read(end-start+2)
                if len(data) != end-start+1:
                    raise ValueError("Truncated range")
                with lock:
                    with temporary.open("r+b") as stream:
                        stream.seek(start)
                        stream.write(data)
                    completed.add(index)
                    write_json(checkpoint, {"sha256": expected, "size": size, "completed": sorted(completed)})
                    if len(completed) % 20 == 0 or len(completed) == total:
                        print(f"{destination.name}: {len(completed)}/{total} ranges", flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt+1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(fetch, i) for i in range(total) if i not in completed]
        for job in as_completed(jobs):
            job.result()
    if file_sha256(temporary) != expected:
        raise ValueError("Downloaded artifact differs from pinned SHA256")
    temporary.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"], required=True)
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    parser.add_argument("--probe", action="store_true", help="Read only a small weight range and exit")
    parser.add_argument("--revision", help="Explicitly override the project's frozen model revision")
    args = parser.parse_args()
    api = HfApi()
    info = api.model_info(args.model, revision=args.revision or MODEL_REVISIONS[args.model], files_metadata=True)
    names = [s.rfilename for s in info.siblings]
    patterns = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "sentencepiece.bpe.model", "README.md", "modules.json", "sentence_bert_config.json", "1_Pooling/config.json"]
    if not args.tokenizer_only:
        # Never fetch both PyTorch and safetensors copies of the same weights.
        patterns += ["model.safetensors"] if "model.safetensors" in names else ["pytorch_model.bin"]
        if args.model.endswith("bge-m3"):
            patterns += ["colbert_linear.pt", "sparse_linear.pt"]
    target = Path("models")/args.model.split("/")[1]
    print(f"Downloading {args.model} revision {info.sha} to {target}", flush=True)
    if args.source == "huggingface":
        snapshot_download(args.model, revision=info.sha, local_dir=str(target), allow_patterns=patterns, max_workers=2)
    else:
        # BAAI's alternative distribution; require equality to official HF SHA256.
        target.mkdir(parents=True, exist_ok=True)
        for filename in patterns:
            entry = next((s for s in info.siblings if s.rfilename == filename), None)
            if entry is None:
                continue
            expected = getattr(entry.lfs, "sha256", None) if entry.lfs else None
            if not expected or entry.size < 100_000_000:
                snapshot_download(args.model, revision=info.sha, local_dir=str(target), allow_patterns=[filename])
                continue
            destination = target/filename
            if destination.exists() and file_sha256(destination) == expected:
                continue
            temporary = destination.with_suffix(destination.suffix + ".modelscope-download")
            url = f"https://modelscope.cn/models/{args.model}/resolve/master/{filename}"
            if args.probe:
                started = time.perf_counter()
                request = urllib.request.Request(url, headers={"Range": "bytes=0-65535"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    print({"status": response.status, "content_length": response.headers.get("Content-Length"),
                           "content_range": response.headers.get("Content-Range")}, flush=True)
                    block = response.read(65536)
                    print({"bytes": len(block), "seconds": time.perf_counter()-started}, flush=True)
                raise SystemExit(0)
            print(f"Downloading {filename} from BAAI ModelScope; verifying SHA256 {expected}", flush=True)
            ranged_download(url, destination, entry.size, expected)
    write_json(target/"provenance.json", {"repository": args.model, "revision": info.sha,
               "tokenizer_only": args.tokenizer_only, "patterns": patterns, "license": "see model card",
               "download_source": args.source, "large_file_hashes": {s.rfilename: s.lfs.sha256 for s in info.siblings if s.rfilename in patterns and s.lfs}})
    print("Download complete", flush=True)
