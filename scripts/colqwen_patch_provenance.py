"""Record visual-token order for real ColQwen page checkpoints."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from colqwen_encode_candidates import check_marker
from colqwen_smoke import rows, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoded', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-pages', type=int, default=100000)
    args = p.parse_args()
    if args.max_pages < 1:
        raise ValueError('Invalid page bound')
    from PIL import Image
    from colpali_engine.models import ColQwen2Processor
    source, output = Path(args.encoded), Path(args.output)
    contract = json.loads((source / 'contract.json').read_text(encoding='utf-8'))
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'colpali-engine', 'tokenizers')}
    if versions != contract['runtime_versions']:
        raise ValueError('Processor runtime differs from the encoding contract')
    if sha(Path(args.adapter) / 'adapter_model.safetensors') != contract['weight_sha256']['adapter_model.safetensors']:
        raise ValueError('Adapter weights differ from the encoding contract')
    prepared = Path('data/raw/irpapers-prepared-20261106')
    if sha(prepared / 'pages.jsonl') != contract['source_pages_sha256']:
        raise ValueError('Source pages changed')
    pages = {r['page_id']: r for r in rows(prepared / 'pages.jsonl')}
    processor = ColQwen2Processor.from_pretrained(args.adapter, local_files_only=True,
                                                  max_num_visual_tokens=contract['max_visual_tokens'])
    output.mkdir(parents=True, exist_ok=True)
    produced = 0
    for marker in sorted((source / 'pages').glob('*.json')):
        page_id = marker.stem
        page = pages[page_id]
        expected = {'page_id': page_id, 'paper_id': page['paper_id'], 'image_sha256': page['image_sha256']}
        if not check_marker(source / 'pages', page_id, expected):
            raise ValueError('Missing encoded page checkpoint')
        encoded = json.loads(marker.read_text(encoding='utf-8'))
        target = output / (page_id + '.json')
        if target.exists():
            previous = json.loads(target.read_text(encoding='utf-8'))
            if (previous.get('encoded_marker_sha256') != sha(marker)
                    or previous.get('encoding_contract_sha256') != sha(source / 'contract.json')):
                raise ValueError('Existing patch provenance does not match source')
            continue
        if produced >= args.max_pages:
            break
        path = prepared / page['image_path']
        if sha(path) != page['image_sha256']:
            raise ValueError('Page image changed')
        with Image.open(path) as image:
            rgb = image.convert('RGB')
        batch = processor.process_images([rgb])
        rgb.close()
        active = batch['attention_mask'][0].bool()
        image_mask = processor.get_image_mask(batch)[0][active]
        positions = image_mask.nonzero(as_tuple=True)[0].tolist()
        grid = batch['image_grid_thw'].tolist()
        if grid != encoded['image_grid_thw'] or int(active.sum()) != encoded['token_count']:
            raise ValueError('Processor grid or active-token count changed')
        merge = int(processor.image_processor.merge_size)
        t, h, w = grid[0]
        if h % merge or w % merge or len(positions) != t * (h // merge) * (w // merge):
            raise ValueError('Visual token positions do not match the processor grid')
        result = {**expected, 'encoding_contract_sha256': sha(source / 'contract.json'),
                  'encoded_marker_sha256': sha(marker), 'vector_file_sha256': encoded['vectors_sha256'],
                  'image_grid_thw': grid, 'spatial_merge_size': merge, 'merged_grid_thw': [t, h // merge, w // merge],
                  'active_token_count': int(active.sum()), 'image_token_positions': positions,
                  'order': 'model input image-token sequence; no pixel-level box claim'}
        with target.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True)
        produced += 1
        if produced % 20 == 0:
            print('patch page records written', produced, flush=True)
    print(json.dumps({'new_pages': produced, 'available_checkpoints': len(list((source / 'pages').glob('*.json')))}))


if __name__ == '__main__': main()
