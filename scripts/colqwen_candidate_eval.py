"""Resume real ColQwen candidates into Chroma and rerank within that scope."""
import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from colqwen_encode_candidates import check_marker
from colqwen_smoke import rows, sha
from irpapers_optimize import read_frozen
from paper_agent.resumable_colqwen import ResumableColQwenStore
from paper_agent.colqwen_store import maxsim
from paper_agent.maxsim_batch import score_maxsim_batch


def read_vector(folder, key, expected):
    if not check_marker(folder, key, expected):
        return None
    metadata = json.loads((folder / (key + '.json')).read_text(encoding='utf-8'))
    with np.load(folder / metadata['vector_file'], allow_pickle=False) as data:
        vector = np.asarray(data['vectors'], dtype=np.float32)
    if vector.shape != (metadata['token_count'], metadata['dimension']):
        raise ValueError('Checkpoint vector shape mismatch')
    return vector


def install_patch_provenance(index, source, provenance_dir, page_ids, pages):
    if provenance_dir is None:
        raise ValueError('Patch provenance is required before sealing candidate pages')
    provenance_dir = Path(provenance_dir)
    lines = []
    contract_hash = sha(source / 'contract.json')
    for page_id in page_ids:
        marker = source / 'pages' / (page_id + '.json')
        path = provenance_dir / (page_id + '.json')
        if not path.is_file():
            raise ValueError('Patch provenance missing for page ' + page_id)
        data = json.loads(path.read_text(encoding='utf-8'))
        encoded = json.loads(marker.read_text(encoding='utf-8'))
        positions = data.get('image_token_positions')
        grid = data.get('merged_grid_thw')
        raw_grid = data.get('image_grid_thw')
        merge = data.get('spatial_merge_size')
        count = data.get('active_token_count')
        if (data.get('page_id') != page_id or data.get('paper_id') != pages[page_id]['paper_id']
                or data.get('image_sha256') != pages[page_id]['image_sha256']
                or data.get('encoding_contract_sha256') != contract_hash or data.get('encoded_marker_sha256') != sha(marker)
                or data.get('vector_file_sha256') != encoded['vectors_sha256'] or data.get('image_grid_thw') != encoded['image_grid_thw']
                or type(count) is not int or count < 1 or count != encoded['token_count']
                or not isinstance(positions, list) or not isinstance(grid, list) or len(grid) != 3
                or any(type(x) is not int or x < 1 for x in grid)
                or type(merge) is not int or merge < 1 or not isinstance(raw_grid, list) or len(raw_grid) != 1
                or not isinstance(raw_grid[0], list) or len(raw_grid[0]) != 3
                or any(type(x) is not int or x < 1 for x in raw_grid[0])
                or raw_grid[0][1] % merge or raw_grid[0][2] % merge
                or grid != [raw_grid[0][0], raw_grid[0][1] // merge, raw_grid[0][2] // merge]
                or any(type(i) is not int or i < 0 or i >= count for i in positions)
                or positions != sorted(set(positions)) or len(positions) != grid[0] * grid[1] * grid[2]):
            raise ValueError('Invalid patch provenance for page ' + page_id)
        lines.append(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    contents = ''.join(lines).encode('utf-8')
    fingerprint = hashlib.sha256(contents).hexdigest()
    target = index / 'patch-provenance.jsonl'
    if target.exists():
        if sha(target) != fingerprint:
            raise ValueError('Existing patch provenance differs; refusing to overwrite')
    else:
        partial = index / ('patch-provenance.' + uuid.uuid4().hex + '.pending')
        with partial.open('xb') as stream:
            stream.write(contents)
        if target.exists():
            raise ValueError('Patch provenance appeared concurrently; pending file preserved')
        partial.rename(target)
    return fingerprint


def load_scoring_checkpoint(path, signature, query_hash, pages_hash, case_id):
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding='utf-8'))
    record = saved.get('result')
    record_hash = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if (saved.get('signature') != signature or saved.get('query_marker_sha256') != query_hash
            or saved.get('candidate_pages_sha256') != pages_hash or saved.get('result_sha256') != record_hash
            or not isinstance(record, dict) or record.get('case_id') != case_id):
        raise ValueError('Scoring checkpoint mismatch: ' + case_id)
    return saved


def save_scoring_checkpoint(path, signature, query_hash, pages_hash, reference_difference, record):
    envelope = {'signature': signature, 'query_marker_sha256': query_hash, 'candidate_pages_sha256': pages_hash,
                'reference_max_abs_difference': reference_difference, 'result': record,
                'result_sha256': hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    ResumableColQwenStore._write_new(path, envelope)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoded', required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--selection')
    p.add_argument('--patch-provenance')
    p.add_argument('--score-device', default='cuda:0')
    p.add_argument('--cache-pages', type=int, default=256)
    p.add_argument('--max-pages', type=int, default=100000)
    p.add_argument('--score-max-seconds', type=int, default=600)
    p.add_argument('--max-queries', type=int, default=100000)
    args = p.parse_args()
    if args.max_pages < 1 or args.score_max_seconds < 1 or args.max_queries < 1 or not 1 <= args.cache_pages <= 4096:
        raise ValueError('Invalid page bound')
    import psutil
    source, candidate_dir, root = Path(args.encoded), Path(args.candidates), Path(args.baseline)
    config = json.loads((source / 'contract.json').read_text(encoding='utf-8'))
    if config['partition'] == 'test':
        if not args.selection or sha(args.selection) != config.get('selection_sha256'):
            raise ValueError('Frozen development selection is required for test evaluation')
        selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
        if (selection.get('status') != 'frozen_before_test_evaluation'
                or sha(selection['development_summary']) != selection['development_summary_sha256']
                or selection['encoder_script_sha256'] != config['script_sha256']
                or selection['evaluation_script_sha256'] != sha(__file__)
                or selection['store_code_sha256'] != sha('src/paper_agent/resumable_colqwen.py')
                or selection['maxsim_code_sha256'] != sha('src/paper_agent/colqwen_store.py')
                or selection['maxsim_batch_code_sha256'] != sha('src/paper_agent/maxsim_batch.py')
                or selection['score_device'] != args.score_device or selection['cache_pages'] != args.cache_pages
                or selection['scoring_torch_version'] != importlib.metadata.version('torch')):
            raise ValueError('Development selection or summary changed')
    frozen = read_frozen(root)
    plan = json.loads((candidate_dir / 'plan.json').read_text(encoding='utf-8'))
    candidate_summary = json.loads((candidate_dir / 'results.jsonl.summary.json').read_text(encoding='utf-8'))
    if (config['candidate_plan_sha256'] != sha(candidate_dir / 'plan.json')
            or config['candidate_results_sha256'] != sha(candidate_dir / 'results.jsonl')
            or config['script_sha256'] != sha('scripts/colqwen_encode_candidates.py')
            or candidate_summary['results_sha256'] != config['candidate_results_sha256']
            or plan['baseline_freeze_sha256'] != sha(root / 'BASELINE-FROZEN.json')
            or plan['baseline_files'] != frozen['files']):
        raise ValueError('Candidate or baseline provenance mismatch')
    prepared = Path('data/raw/irpapers-prepared-20261106')
    if sha(prepared / 'pages.jsonl') != config['source_pages_sha256']:
        raise ValueError('Prepared source pages changed')
    pages = {r['page_id']: r for r in rows(prepared / 'pages.jsonl')}
    candidate = [r for r in rows(candidate_dir / 'results.jsonl') if r['partition'] == config['partition']]
    baseline = {r['case_id']: r for r in rows(root / 'text-results.jsonl') if r['partition'] == config['partition']}
    visual = {r['case_id']: r for r in rows(root / 'visual-results.jsonl') if r['partition'] == config['partition']}
    if (len(candidate) != len(baseline) or {r['case_id'] for r in candidate} != set(baseline)
            or set(visual) != set(baseline)):
        raise ValueError('Candidate query identities differ')
    page_ids = list(dict.fromkeys(page for row in candidate for page in row['pages']))
    contract = {'model_revision': config['model_revision'], 'base_revision': config['base_revision'],
                'source_hash': config['source_pages_sha256'], 'dimension': config['dimension'],
                'partition': config['partition'], 'candidate_plan_sha256': config['candidate_plan_sha256'],
                'candidate_results_sha256': config['candidate_results_sha256'], 'smoke_sha256': config['smoke_sha256'],
                'max_visual_tokens': config['max_visual_tokens'], 'weight_sha256': config['weight_sha256']}
    index = Path(args.index)
    store = ResumableColQwenStore(index, contract, create=not index.exists())
    if store.final.exists() and not json.loads(store.final.read_text(encoding='utf-8')).get('patch_provenance_sha256'):
        raise ValueError('Sealed candidate index lacks patch provenance')
    start = time.perf_counter()
    process = psutil.Process()
    sampled_rss = process.memory_info().rss
    added = 0
    missing = []
    if not store.final.exists():
        for page_id in page_ids:
            page = pages[page_id]
            vector = read_vector(source / 'pages', page_id, {'page_id': page_id, 'paper_id': page['paper_id'],
                                                              'image_sha256': page['image_sha256']})
            if vector is None:
                missing.append(page_id)
                continue
            if sha(prepared / page['image_path']) != page['image_sha256']:
                raise ValueError('Source page image changed')
            if not (store.committed / (page_id + '.json')).exists():
                if added >= args.max_pages:
                    missing.append(page_id)
                    continue
                store.add_page(page_id, page['paper_id'], page['image_sha256'], vector)
                added += 1
                if added % 20 == 0:
                    sampled_rss = max(sampled_rss, process.memory_info().rss)
                    print('pages committed', added, 'of', len(page_ids), flush=True)
        if missing:
            attempt = {'status': 'incomplete', 'new_pages_committed': added, 'remaining_pages': len(missing),
                       'committed_tokens': store.collection.count(), 'elapsed_seconds': time.perf_counter() - start,
                       'sampled_peak_rss_bytes': max(sampled_rss, process.memory_info().rss),
                       'index_bytes': sum(p.stat().st_size for p in index.rglob('*') if p.is_file())}
            attempts = index / 'import-attempts'
            attempts.mkdir(exist_ok=True)
            with (attempts / (uuid.uuid4().hex + '.json')).open('x', encoding='utf-8') as stream:
                json.dump(attempt, stream, indent=2)
            print(json.dumps(attempt))
            return
        provenance_hash = install_patch_provenance(index, source, args.patch_provenance, page_ids, pages)
        store.seal(patch_provenance_sha256=provenance_hash)
    output = Path(args.output)
    if output.exists() or Path(str(output) + '.summary.json').exists():
        raise ValueError('Refusing to overwrite existing evaluation output')
    for source_index, page_id in enumerate(page_ids, 1):
        if sha(prepared / pages[page_id]['image_path']) != pages[page_id]['image_sha256']:
            raise ValueError('Source page image changed: ' + page_id)
        if source_index % 500 == 0:
            print('source images verified', source_index, 'of', len(page_ids), flush=True)
    import torch
    result = []
    if args.score_device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA scoring device is unavailable')
        torch.cuda.reset_peak_memory_stats()
    cache = OrderedDict()
    verification_max_diff = None
    checkpoint_dir = Path(str(output) + '.checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    spec = {'schema': 1, 'partition': config['partition'], 'index_seal_sha256': sha(index / 'sealed.json'),
            'encoding_contract_sha256': sha(source / 'contract.json'), 'candidate_results_sha256': config['candidate_results_sha256'],
            'evaluation_script_sha256': sha(__file__), 'store_code_sha256': sha('src/paper_agent/resumable_colqwen.py'),
            'maxsim_code_sha256': sha('src/paper_agent/colqwen_store.py'), 'maxsim_batch_code_sha256': sha('src/paper_agent/maxsim_batch.py'),
            'score_device': args.score_device, 'cache_pages': args.cache_pages, 'scoring_torch_version': torch.__version__}
    spec_path = checkpoint_dir / 'contract.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding='utf-8')) != spec:
            raise ValueError('Scoring checkpoint contract differs')
    else:
        ResumableColQwenStore._write_new(spec_path, spec)
    signature = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    score_deadline = time.monotonic() + args.score_max_seconds
    newly_scored = 0
    verified_pages = set()
    for row in candidate:
        original = baseline[row['case_id']]
        old_visual = visual[row['case_id']]
        if any(x['status'] != 'ok' for x in (original['bm25'], original['dense'], original['hybrid'], old_visual['clip'], old_visual['fusion'])):
            raise ValueError('A frozen baseline branch failed')
        query = read_vector(source / 'queries', row['case_id'],
                            {'case_id': row['case_id'], 'question_sha256': hashlib.sha256(original['question'].encode()).hexdigest()})
        if query is None:
            raise ValueError('A candidate query has not been encoded')
        query_marker = source / 'queries' / (row['case_id'] + '.json')
        query_record = json.loads(query_marker.read_text(encoding='utf-8'))
        t = time.perf_counter()
        if len(row['pages']) != len(set(row['pages'])):
            raise ValueError('Duplicate candidate page')
        pages_hash = hashlib.sha256(json.dumps(row['pages'], ensure_ascii=False).encode()).hexdigest()
        checkpoint = checkpoint_dir / (row['case_id'] + '.json')
        saved = load_scoring_checkpoint(checkpoint, signature, sha(query_marker), pages_hash, row['case_id'])
        if saved is not None:
            record = saved['result']
            for page in row['pages']:
                if page not in verified_pages:
                    store.get_page(page)
                    verified_pages.add(page)
            if verification_max_diff is None:
                verification_max_diff = saved.get('reference_max_abs_difference')
            result.append(record)
            if len(result) % 10 == 0:
                print('cases scored or resumed', len(result), 'of', len(candidate), flush=True)
            continue
        if newly_scored >= args.max_queries or time.monotonic() >= score_deadline:
            break
        documents = []
        for page in row['pages']:
            if page in cache:
                cache.move_to_end(page)
            else:
                cache[page] = store.get_page(page)
                verified_pages.add(page)
                if len(cache) > args.cache_pages:
                    cache.popitem(last=False)
            documents.append(cache[page])
        read_seconds = time.perf_counter() - t
        scoring_start = time.perf_counter()
        scores = score_maxsim_batch(query, documents, device=args.score_device)
        scoring_seconds = time.perf_counter() - scoring_start
        if verification_max_diff is None:
            verification_max_diff = max(abs(scores[i] - maxsim(query, documents[i])) for i in range(min(5, len(documents))))
            if verification_max_diff > 1e-3:
                raise ValueError('Batched MaxSim differs from per-page reference')
        reranked = [page for page, _score in sorted(zip(row['pages'], scores), key=lambda item: (-item[1], item[0]))]
        seen = set(reranked)
        fused_tail = reranked + [page for page in original['hybrid']['pages'] if page not in seen]
        gold = original['gold_page']
        valid = original['gold_status'] == 'consistent'
        candidate_rank = reranked.index(gold) + 1 if valid and gold in reranked else None
        full_rank = fused_tail.index(gold) + 1 if valid and gold in fused_tail else None
        record = {'case_id': row['case_id'], 'group': row['group'], 'partition': row['partition'],
                       'gold_status': original['gold_status'], 'candidate_count': len(reranked),
                       'candidate_hit': candidate_rank is not None, 'candidate_rank': candidate_rank,
                       'reranked_with_text_tail_rank': full_rank, 'text_rank': original['hybrid']['gold_rank'],
                       'bm25_rank': original['bm25']['gold_rank'], 'dense_rank': original['dense']['gold_rank'],
                       'clip_rank': old_visual['clip']['gold_rank'], 'equal_fusion_rank': old_visual['fusion']['gold_rank'],
                       'query_encode_seconds': query_record['encode_seconds'],
                       'prior_text_search_seconds': original['hybrid']['elapsed_seconds'],
                       'prior_clip_search_seconds': old_visual['clip']['elapsed_seconds'],
                       'top20': reranked[:20], 'read_seconds': read_seconds, 'score_seconds': scoring_seconds,
                       'rerank_seconds': time.perf_counter() - t, 'sampled_rss_bytes': process.memory_info().rss,
                       'scoring_gpu_peak_bytes': torch.cuda.max_memory_allocated() if args.score_device.startswith('cuda') else None}
        save_scoring_checkpoint(checkpoint, signature, sha(query_marker), pages_hash, verification_max_diff, record)
        result.append(record)
        newly_scored += 1
        if len(result) % 10 == 0:
            print('cases scored or resumed', len(result), 'of', len(candidate), flush=True)
        sampled_rss = max(sampled_rss, process.memory_info().rss)
    if len(result) != len(candidate):
        print(json.dumps({'status': 'scoring_paused', 'scored_or_resumed': len(result),
                          'remaining_cases': len(candidate) - len(result), 'newly_scored': newly_scored}), flush=True)
        return
    if (sha(__file__) != spec['evaluation_script_sha256']
            or sha('src/paper_agent/resumable_colqwen.py') != spec['store_code_sha256']
            or sha('src/paper_agent/colqwen_store.py') != spec['maxsim_code_sha256']
            or sha('src/paper_agent/maxsim_batch.py') != spec['maxsim_batch_code_sha256']):
        raise ValueError('Evaluation implementation changed during scoring')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        for row in result: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = {}
    for group in ('standard', 'visual'):
        subset = [r for r in result if r['group'] == group]
        n = len(subset)
        if not n: continue
        stats = {'n': n, 'candidate_coverage': sum(r['candidate_hit'] for r in subset) / n}
        for key in ('bm25_rank', 'dense_rank', 'text_rank', 'clip_rank', 'equal_fusion_rank', 'reranked_with_text_tail_rank'):
            ranks = [r[key] for r in subset]
            stats[key] = {'hit1': sum(r is not None and r <= 1 for r in ranks) / n,
                          'hit5': sum(r is not None and r <= 5 for r in ranks) / n,
                          'hit20': sum(r is not None and r <= 20 for r in ranks) / n,
                          'mrr': sum(1 / r if r else 0 for r in ranks) / n}
        for k in (1, 5, 20):
            hit = lambda r, key: r[key] is not None and r[key] <= k
            stats[f'paired_hit{k}'] = {'improved': [r['case_id'] for r in subset if hit(r, 'reranked_with_text_tail_rank') and not hit(r, 'text_rank')],
                                       'regressed': [r['case_id'] for r in subset if hit(r, 'text_rank') and not hit(r, 'reranked_with_text_tail_rank')]}
        stats['rerank_median_seconds'] = statistics.median(r['rerank_seconds'] for r in subset)
        stats['query_encode_median_seconds'] = statistics.median(r['query_encode_seconds'] for r in subset)
        summary[group] = stats
    report = {'status': 'complete', 'scope': 'MaxSim only within frozen candidate pages; remaining pages retain text order',
              'partition': config['partition'], 'strict_blind_test': False, 'results_sha256': sha(output),
              'encoding_contract_sha256': sha(source / 'contract.json'),
              'selection_sha256': config.get('selection_sha256'), 'candidate_plan_sha256': config['candidate_plan_sha256'],
              'evaluation_script_sha256': sha(__file__), 'store_code_sha256': sha('src/paper_agent/resumable_colqwen.py'),
              'maxsim_code_sha256': sha('src/paper_agent/colqwen_store.py'), 'maxsim_batch_code_sha256': sha('src/paper_agent/maxsim_batch.py'),
              'score_device': args.score_device, 'cache_pages': args.cache_pages, 'scoring_torch_version': torch.__version__,
              'batch_reference_max_abs_difference': verification_max_diff,
              'peak_gpu_allocated_bytes': max(r['scoring_gpu_peak_bytes'] for r in result) if args.score_device.startswith('cuda') else None,
              'metrics': summary, 'elapsed_seconds': time.perf_counter() - start,
              'recorded_rerank_seconds_total': sum(r['rerank_seconds'] for r in result),
              'sampled_peak_rss_bytes': max([sampled_rss] + [r['sampled_rss_bytes'] for r in result]),
              'index_bytes': sum(p.stat().st_size for p in index.rglob('*') if p.is_file()),
              'latency_scope': 'query encoding and candidate search were measured separately, not a live end-to-end latency'}
    with Path(str(output) + '.summary.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
