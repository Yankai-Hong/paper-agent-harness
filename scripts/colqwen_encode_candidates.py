"""Checkpoint real ColQwen page/query vectors for frozen IRPAPERS candidates."""
import argparse
import importlib.metadata
import json
import re
import time
import uuid
from pathlib import Path

from colqwen_smoke import rows, sha
from irpapers_optimize import read_frozen


def check_marker(folder, key, expected):
    marker = folder / (key + '.json')
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding='utf-8'))
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint metadata mismatch: ' + key)
    filename = record.get('vector_file', '')
    if Path(filename).name != filename or not filename.startswith(key + '-') or not filename.endswith('.npz'):
        raise ValueError('Invalid checkpoint vector path')
    path = folder / filename
    if not path.is_file() or sha(path) != record.get('vectors_sha256'):
        raise ValueError('Checkpoint vector content mismatch: ' + key)
    return True


def save_vector(folder, key, vector, metadata):
    import numpy as np
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', key):
        raise ValueError('Invalid checkpoint identity')
    vector = np.asarray(vector, dtype=np.float32)
    if (vector.ndim != 2 or not vector.shape[0] or not np.isfinite(vector).all()
            or not np.allclose(np.linalg.norm(vector, axis=1), 1, atol=2e-3)):
        raise ValueError('Model returned invalid token vectors')
    filename = key + '-' + uuid.uuid4().hex + '.npz'
    path = folder / filename
    with path.open('xb') as stream:
        np.savez_compressed(stream, vectors=vector)
    record = {**metadata, 'token_count': len(vector), 'dimension': vector.shape[1],
              'vector_file': filename, 'vectors_sha256': sha(path)}
    with (folder / (key + '.json')).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, sort_keys=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--smoke', required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--partition', choices=['dev', 'test'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--selection')
    p.add_argument('--max-seconds', type=int, default=600)
    p.add_argument('--max-pages', type=int, default=100000)
    args = p.parse_args()
    if args.max_seconds < 1 or args.max_pages < 1:
        raise ValueError('Invalid runtime bounds')
    smoke_dir = Path(args.smoke)
    smoke = json.loads((smoke_dir / 'result.json').read_text(encoding='utf-8'))
    if (smoke.get('status') != 'ok' or not smoke.get('maxsim_checked_against_processor')
            or smoke.get('vectors_sha256') != sha(smoke_dir / 'vectors.npz')):
        raise ValueError('A verified real smoke run is required')
    base, adapter = Path(args.base), Path(args.adapter)
    for filename, expected in smoke['weight_sha256'].items():
        path = (adapter if filename == 'adapter_model.safetensors' else base) / filename
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Current model weights do not match smoke run')
    root = Path(args.baseline)
    frozen = read_frozen(root)
    candidate_dir = Path(args.candidates)
    candidate_summary = json.loads((candidate_dir / 'results.jsonl.summary.json').read_text(encoding='utf-8'))
    plan = json.loads((candidate_dir / 'plan.json').read_text(encoding='utf-8'))
    if (candidate_summary['results_sha256'] != sha(candidate_dir / 'results.jsonl')
            or candidate_summary['plan_sha256'] != sha(candidate_dir / 'plan.json')
            or plan['baseline_freeze_sha256'] != sha(root / 'BASELINE-FROZEN.json')
            or plan['baseline_files'] != frozen['files']):
        raise ValueError('Frozen candidate or baseline mismatch')
    selection_hash = None
    if args.partition == 'test':
        if not args.selection:
            raise ValueError('Frozen development selection is required for test encoding')
        selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
        if (selection.get('status') != 'frozen_before_test_evaluation'
                or sha(selection['development_summary']) != selection['development_summary_sha256']
                or selection['candidate_plan_sha256'] != sha(candidate_dir / 'plan.json')
                or selection['candidate_results_sha256'] != sha(candidate_dir / 'results.jsonl')
                or any(selection[k] != smoke[k] for k in ('model_revision', 'base_revision', 'weight_sha256', 'max_visual_tokens', 'dimension'))):
            raise ValueError('Frozen development selection differs from test settings')
        selection_hash = sha(args.selection)
    candidates = [r for r in rows(candidate_dir / 'results.jsonl') if r['partition'] == args.partition]
    baseline = {r['case_id']: r for r in rows(root / 'text-results.jsonl') if r['partition'] == args.partition}
    if len(candidates) != len(baseline) or {r['case_id'] for r in candidates} != set(baseline):
        raise ValueError('Candidate query identities differ')
    page_ids = list(dict.fromkeys(page for r in candidates for page in r['pages']))
    prepared = Path('data/raw/irpapers-prepared-20261106')
    pages = {r['page_id']: r for r in rows(prepared / 'pages.jsonl')}
    if not set(page_ids).issubset(pages):
        raise ValueError('Candidate source page missing')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'colpali-engine', 'tokenizers')}
    if versions['torch'] != smoke['torch']:
        raise ValueError('PyTorch version differs from the verified smoke')
    if args.partition == 'test' and (selection['runtime_versions'] != versions or selection['device'] != smoke['device']):
        raise ValueError('Test runtime differs from the frozen development selection')
    contract = {'schema': 1, 'partition': args.partition, 'smoke_sha256': sha(smoke_dir / 'result.json'),
                'candidate_plan_sha256': sha(candidate_dir / 'plan.json'), 'candidate_results_sha256': sha(candidate_dir / 'results.jsonl'),
                'source_pages_sha256': sha(prepared / 'pages.jsonl'), 'script_sha256': sha(__file__),
                'model_revision': smoke['model_revision'], 'base_revision': smoke['base_revision'],
                'weight_sha256': smoke['weight_sha256'], 'max_visual_tokens': smoke['max_visual_tokens'],
                'dimension': smoke['dimension'], 'device': smoke['device'], 'selection_sha256': selection_hash,
                'runtime_versions': versions}
    out = Path(args.output)
    if out.exists():
        if json.loads((out / 'contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Cannot resume with a changed encoding contract')
    else:
        out.mkdir(parents=True)
        (out / 'pages').mkdir()
        (out / 'queries').mkdir()
        with (out / 'contract.json').open('x', encoding='utf-8') as stream:
            json.dump(contract, stream, ensure_ascii=False, sort_keys=True, indent=2)
    (out / 'attempts').mkdir(exist_ok=True)
    missing_pages = [page for page in page_ids if not check_marker(out / 'pages', page,
        {'page_id': page, 'image_sha256': pages[page]['image_sha256']})]
    missing_queries = [r['case_id'] for r in candidates if not check_marker(out / 'queries', r['case_id'],
        {'case_id': r['case_id'], 'question_sha256': __import__('hashlib').sha256(baseline[r['case_id']]['question'].encode()).hexdigest()})]
    if not missing_pages and not missing_queries:
        print(json.dumps({'status': 'complete', 'pages': len(page_ids), 'queries': len(candidates)}))
        return
    import hashlib
    import psutil
    import torch
    from PIL import Image
    from peft import PeftModel
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    device = smoke['device']
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processor = ColQwen2Processor.from_pretrained(str(adapter), local_files_only=True,
                                                  max_num_visual_tokens=smoke['max_visual_tokens'])
    if processor.query_prefix != 'Query: ':
        raise ValueError('Unexpected query prefix')
    model = ColQwen2.from_pretrained(str(base), torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
                                     device_map=device, local_files_only=True, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True).eval()
    deadline = time.monotonic() + args.max_seconds
    for query_index, case_id in enumerate(missing_queries, 1):
        if time.monotonic() >= deadline:
            break
        item_started = time.perf_counter()
        question = baseline[case_id]['question']
        batch = processor.process_queries([question]).to(device)
        with torch.inference_mode():
            vectors = model(**batch)[0].float().cpu().numpy()
        vectors = vectors[batch['attention_mask'][0].cpu().numpy().astype(bool)]
        if vectors.shape[1] != smoke['dimension']:
            raise ValueError('Query embedding dimension differs from smoke')
        save_vector(out / 'queries', case_id, vectors,
                    {'case_id': case_id, 'question_sha256': hashlib.sha256(question.encode()).hexdigest(),
                     'encode_seconds': time.perf_counter() - item_started})
        if query_index % 20 == 0:
            print('queries encoded this run', query_index, 'of', len(missing_queries), flush=True)
    completed = 0
    for page_id in missing_pages:
        if time.monotonic() >= deadline or completed >= args.max_pages:
            break
        item_started = time.perf_counter()
        page = pages[page_id]
        path = prepared / page['image_path']
        if sha(path) != page['image_sha256']:
            raise ValueError('Candidate image hash mismatch')
        with Image.open(path) as image:
            im = image.convert('RGB')
        batch = processor.process_images([im]).to(device)
        im.close()
        with torch.inference_mode():
            vectors = model(**batch)[0].float().cpu().numpy()
        vectors = vectors[batch['attention_mask'][0].cpu().numpy().astype(bool)]
        if vectors.shape[1] != smoke['dimension']:
            raise ValueError('Page embedding dimension differs from smoke')
        save_vector(out / 'pages', page_id, vectors,
                    {'page_id': page_id, 'paper_id': page['paper_id'], 'image_sha256': page['image_sha256'],
                     'image_grid_thw': batch['image_grid_thw'].cpu().tolist(),
                     'encode_seconds': time.perf_counter() - item_started})
        completed += 1
        if completed % 20 == 0:
            print('pages encoded this run', completed, 'of', len(missing_pages), flush=True)
    remaining_pages = sum(not (out / 'pages' / (page + '.json')).exists() for page in page_ids)
    remaining_queries = sum(not (out / 'queries' / (r['case_id'] + '.json')).exists() for r in candidates)
    report = {'status': 'complete' if not remaining_pages and not remaining_queries else 'paused',
              'remaining_pages': remaining_pages, 'remaining_queries': remaining_queries,
              'elapsed_seconds': time.perf_counter() - started, 'new_pages': completed,
              'rss_bytes_final': psutil.Process().memory_info().rss,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else None}
    with (out / 'attempts' / (uuid.uuid4().hex + '.json')).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
