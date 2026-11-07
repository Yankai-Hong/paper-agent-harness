"""Encode verified Docling PDF pages as resumable real ColQwen token vectors."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from colqwen_encode_candidates import check_marker, save_vector
from colqwen_smoke import rows, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared', 'reference-contract', 'base', 'adapter', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    import psutil
    import torch
    from PIL import Image
    from paper_agent.colqwen_query_encoder import ColQwenQueryEncoder
    source = Path(a.prepared)
    binding = json.loads((source / 'binding.json').read_text(encoding='utf-8'))
    pages_path = source / 'pages.jsonl'
    if binding.get('status') != 'sealed' or binding.get('pages_sha256') != sha(pages_path):
        raise ValueError('Prepared PDF source mismatch')
    reference = json.loads(Path(a.reference_contract).read_text(encoding='utf-8'))
    contract = {key: reference[key] for key in ('model_revision', 'base_revision', 'weight_sha256',
                                                 'max_visual_tokens', 'dimension', 'smoke_sha256')}
    contract.update(source_hash=sha(pages_path), partition='docling_pdf', device=a.device)
    output = Path(a.output)
    if output.exists():
        if json.loads((output / 'contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Cannot resume with a changed PDF encoding contract')
    else:
        output.mkdir(parents=True)
        (output / 'pages').mkdir()
        with (output / 'contract.json').open('x', encoding='utf-8') as stream:
            json.dump(contract, stream, sort_keys=True, indent=2)
    pages = rows(pages_path)
    missing = [page for page in pages if not check_marker(output / 'pages', page['page_id'],
        {'page_id': page['page_id'], 'paper_id': page['paper_id'], 'image_sha256': page['image_sha256']})]
    if not missing:
        print(json.dumps({'status': 'complete', 'pages': len(pages), 'new_pages': 0}))
        return
    if a.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoder = ColQwenQueryEncoder(a.base, a.adapter, contract, device=a.device)
    for count, page in enumerate(missing, 1):
        image_path = (source / page['image_path']).resolve()
        if not image_path.is_relative_to(source.resolve()) or sha(image_path) != page['image_sha256']:
            raise ValueError('Rendered PDF page changed')
        with Image.open(image_path) as image:
            rgb = image.convert('RGB')
        vectors, grid = encoder.encode_page(rgb)
        rgb.close()
        save_vector(output / 'pages', page['page_id'], vectors,
                    {'page_id': page['page_id'], 'paper_id': page['paper_id'],
                     'image_sha256': page['image_sha256'], **grid})
        if count % 10 == 0:
            print('PDF pages encoded', count, 'of', len(missing), flush=True)
    report = {'status': 'complete', 'pages': len(pages), 'new_pages': len(missing),
              'elapsed_seconds': time.perf_counter() - started,
              'rss_bytes_final': psutil.Process().memory_info().rss,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None}
    with (output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
