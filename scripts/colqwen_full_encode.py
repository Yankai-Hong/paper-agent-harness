"""Build resumable full-corpus page checkpoints, reusing verified real vectors."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from colqwen_encode_candidates import check_marker, save_vector
from colqwen_smoke import rows, sha

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'dev', 'test', 'dev-index', 'test-index', 'base', 'adapter', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seconds', type=int, default=1800)
    a = p.parse_args()
    source, out = Path(a.prepared), Path(a.output)
    pages = rows(source / 'pages.jsonl')
    if len(pages) != 3230 or len({x['page_id'] for x in pages}) != 3230:
        raise ValueError('Expected 3230 unique source pages')
    page_hash = sha(source / 'pages.jsonl')
    dev, test = Path(a.dev), Path(a.test)
    c1 = json.loads((dev / 'contract.json').read_text(encoding='utf-8'))
    c2 = json.loads((test / 'contract.json').read_text(encoding='utf-8'))
    fields = ('base_revision','model_revision','weight_sha256','max_visual_tokens','dimension',
              'runtime_versions','device','script_sha256','source_pages_sha256','smoke_sha256')
    if any(c1[k] != c2[k] for k in fields) or c1['source_pages_sha256'] != page_hash or c1['device'] != a.device:
        raise ValueError('Source or encoder contracts differ')
    provenance = {}
    for folder, index in ((dev, Path(a.dev_index)), (test, Path(a.test_index))):
        sealed = json.loads((index / 'sealed.json').read_text(encoding='utf-8'))
        if sha(index / 'patch-provenance.jsonl') != sealed['patch_provenance_sha256']:
            raise ValueError('Sealed provenance mismatch')
        entries = rows(index / 'patch-provenance.jsonl')
        provenance[str(folder)] = {x['page_id']:x for x in entries}
        if len(entries) != sealed['pages'] or len(provenance[str(folder)]) != len(entries):
            raise ValueError('Sealed page provenance count mismatch')
    contract = {'schema': 1, 'source_pages_sha256': page_hash, 'encoder': {k:c1[k] for k in fields},
                'dev_contract_sha256': sha(dev / 'contract.json'), 'test_contract_sha256': sha(test / 'contract.json'),
                'dev_seal_sha256': sha(Path(a.dev_index) / 'sealed.json'), 'test_seal_sha256': sha(Path(a.test_index) / 'sealed.json'),
                'implementation_sha256': sha(__file__), 'scope': 'all 3230 pages, no candidate filter'}
    if out.exists():
        if json.loads((out / 'contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Full corpus contract changed')
    else:
        out.mkdir(parents=True); (out / 'pages').mkdir(); (out / 'queries').mkdir()
        with (out / 'contract.json').open('x', encoding='utf-8') as f: json.dump(contract, f, indent=2, sort_keys=True)
    reused = overlap = 0
    missing = []
    for page in pages:
        key = page['page_id']; expected = {'page_id':key,'paper_id':page['paper_id'],'image_sha256':page['image_sha256']}
        if sha(source / page['image_path']) != page['image_sha256']:
            raise ValueError('Source image changed: ' + key)
        available = [folder for folder in (dev, test) if check_marker(folder / 'pages', key, expected)]
        for folder in available:
            record = provenance[str(folder)].get(key)
            marker_path = folder / 'pages' / (key + '.json')
            marker = json.loads(marker_path.read_text(encoding='utf-8'))
            if (not record or record.get('encoded_marker_sha256') != sha(marker_path)
                    or record.get('encoding_contract_sha256') != sha(folder / 'contract.json')
                    or record.get('vector_file_sha256') != marker['vectors_sha256']
                    or record.get('image_sha256') != page['image_sha256']
                    or record.get('image_grid_thw') != marker['image_grid_thw']):
                raise ValueError('Page provenance differs: ' + key)
        if len(available) == 2:
            arrays = []
            for folder in available:
                m = json.loads((folder / 'pages' / (key + '.json')).read_text(encoding='utf-8'))
                with np.load(folder / 'pages' / m['vector_file'], allow_pickle=False) as z: arrays.append(z['vectors'])
            if arrays[0].shape != arrays[1].shape or not np.allclose(arrays[0], arrays[1], atol=1e-6, rtol=0):
                raise ValueError('Overlapping page vectors differ: ' + key)
            overlap += 1
        if check_marker(out / 'pages', key, expected): continue
        if not available:
            missing.append(page); continue
        folder = available[0]; marker = json.loads((folder / 'pages' / (key + '.json')).read_text(encoding='utf-8'))
        target = out / 'pages' / marker['vector_file']
        if not target.exists():
            with (folder / 'pages' / marker['vector_file']).open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
        if sha(target) != marker['vectors_sha256']: raise ValueError('Copied vector differs')
        with (out / 'pages' / (key + '.json')).open('x', encoding='utf-8') as f: json.dump(marker, f, sort_keys=True)
        reused += 1
    print(json.dumps({'reused_this_run':reused,'overlap_checked':overlap,'missing_to_encode':len(missing)}), flush=True)
    if missing:
        import torch
        from PIL import Image
        from paper_agent.colqwen_query_encoder import ColQwenQueryEncoder
        cfg = {k:c1[k] for k in ('model_revision','base_revision','weight_sha256','max_visual_tokens','dimension')}
        encoder = ColQwenQueryEncoder(a.base, a.adapter, cfg, device=a.device)
        deadline = time.monotonic() + a.max_seconds
        for i, page in enumerate(missing, 1):
            if time.monotonic() >= deadline: break
            with Image.open(source / page['image_path']) as image: rgb = image.convert('RGB')
            vector, info = encoder.encode_page(rgb); rgb.close()
            save_vector(out / 'pages', page['page_id'], vector,
                        {'page_id':page['page_id'],'paper_id':page['paper_id'],'image_sha256':page['image_sha256'], **info})
            if i % 50 == 0: print('new pages encoded', i, 'of', len(missing), flush=True)
    remaining = sum(not (out / 'pages' / (x['page_id'] + '.json')).exists() for x in pages)
    print(json.dumps({'status':'complete' if not remaining else 'paused','remaining_pages':remaining}), flush=True)


if __name__ == '__main__': main()
