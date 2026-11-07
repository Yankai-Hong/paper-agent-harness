"""Bounded, verified HTTPS range download of the two pinned official shards.

Partial or previously assembled files are never overwritten or deleted. Each
finished range has a checksum sidecar. A whole shard is published only after
its official SHA-256 has been verified.
"""
import argparse
import concurrent.futures
import hashlib
import json
import time
import uuid
from pathlib import Path

import requests

REVISION = '9fe8a713422a7cb4ef79ca77a09b381ee2243101'
SHARDS = [
    ('model-00001-of-00002.safetensors', 4982048792, '5afb3798d74668030808362aa4a4870cf71a0cfd1780287c9c55e40fb24e51d4'),
    ('model-00002-of-00002.safetensors', 3854759672, '3da90e07b7b0c19944ee2cdcf537fa62ba5b01a9170b193d6b68353e9b033ec3'),
]
CHUNK = 16 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ready(directory, shard, index, start, end):
    prefix = f'{shard:02}-{index:04}-'
    for marker in sorted(directory.glob(prefix + '*.ok.json')):
        meta = json.loads(marker.read_text(encoding='utf-8'))
        filename = meta.get('file', '')
        if (Path(filename).name != filename or not filename.startswith(prefix)
                or not filename.endswith('.part') or marker.name != filename + '.ok.json'):
            raise ValueError('Invalid verified range sidecar path')
        path = directory / filename
        if (meta.get('start') == start and meta.get('end') == end and path.is_file()
                and path.stat().st_size == end - start + 1 and sha(path) == meta.get('sha256')):
            return path
    return None


def fetch(directory, shard, index, start, end):
    name, total, _ = SHARDS[shard]
    url = f'https://huggingface.co/vidore/colqwen2-base/resolve/{REVISION}/{name}'
    tag = uuid.uuid4().hex
    path = directory / f'{shard:02}-{index:04}-{tag}.part'
    expected = f'bytes {start}-{end}/{total}'
    h = hashlib.sha256()
    with requests.get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                      stream=True, timeout=(15, 90)) as response:
        if response.status_code != 206 or response.headers.get('Content-Range') != expected:
            raise ValueError('Official server returned an unexpected range or status')
        with path.open('xb') as stream:
            for block in response.iter_content(1024 * 1024):
                if block:
                    stream.write(block)
                    h.update(block)
    if path.stat().st_size != end - start + 1:
        raise ValueError('Range length mismatch; partial file preserved')
    marker = path.with_name(path.name + '.ok.json')
    with marker.open('x', encoding='utf-8') as stream:
        json.dump({'file': path.name, 'start': start, 'end': end, 'sha256': h.hexdigest()}, stream)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-seconds', type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.max_seconds <= 3600:
        raise ValueError('Invalid bounded download settings')
    directory = Path(args.directory)
    if not directory.is_dir() or not (directory / 'config.json').is_file():
        raise ValueError('Expected fresh official model configuration directory')
    parts = directory / 'verified-ranges'
    parts.mkdir(exist_ok=True)
    pending = []
    found = {}
    for shard, (name, size, expected_hash) in enumerate(SHARDS):
        final = directory / name
        if final.exists():
            if final.stat().st_size != size or sha(final) != expected_hash:
                raise ValueError('Existing shard differs from pinned model; refusing to overwrite')
            continue
        for index, start in enumerate(range(0, size, CHUNK)):
            end = min(size, start + CHUNK) - 1
            key = (shard, index)
            found[key] = ready(parts, shard, index, start, end)
            if found[key] is None:
                pending.append((shard, index, start, end))
    print('verified ranges', sum(x is not None for x in found.values()), 'remaining', len(pending), flush=True)
    deadline = time.monotonic() + args.max_seconds
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        while (pending and time.monotonic() < deadline) or futures:
            while pending and len(futures) < args.workers and time.monotonic() < deadline and not errors:
                task = pending.pop(0)
                futures[pool.submit(fetch, parts, *task)] = task
            if not futures:
                break
            finished, _ = concurrent.futures.wait(futures, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                task = futures.pop(future)
                try:
                    found[task[:2]] = future.result()
                except Exception as exc:
                    errors.append({'shard': task[0], 'range': task[1], 'error_type': type(exc).__name__})
                done = sum(x is not None for x in found.values())
                if done % 8 == 0 or errors:
                    print('verified ranges', done, 'remaining', len(found) - done, 'errors', len(errors), flush=True)
            if errors and not futures:
                break
    if errors:
        print(json.dumps({'status': 'error', 'errors': errors}), flush=True)
        raise SystemExit(1)
    for shard, (name, size, expected_hash) in enumerate(SHARDS):
        final = directory / name
        if final.exists():
            continue
        keys = [(shard, i) for i in range((size + CHUNK - 1) // CHUNK)]
        if not all(found.get(key) is not None for key in keys):
            continue
        temp = directory / (name + '.assembling.' + uuid.uuid4().hex)
        h = hashlib.sha256()
        with temp.open('xb') as out:
            for key in keys:
                with found[key].open('rb') as source:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                        out.write(block)
                        h.update(block)
        if temp.stat().st_size != size or h.hexdigest() != expected_hash:
            raise ValueError('Final official SHA-256 mismatch; assembly preserved')
        if final.exists():
            raise ValueError('Final shard appeared concurrently; assembly preserved')
        temp.rename(final)
        print('verified shard', shard + 1, 'bytes', size, flush=True)
    remaining = sum(not (directory / name).is_file() for name, _, _ in SHARDS)
    print(json.dumps({'status': 'complete' if remaining == 0 else 'paused', 'remaining_shards': remaining}), flush=True)


if __name__ == '__main__':
    main()
