"""Copy verified development page checkpoints into a compatible test run."""
import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from colqwen_encode_candidates import check_marker
from colqwen_smoke import rows, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--prepared', required=True)
    args = p.parse_args()
    dev, test = Path(args.dev), Path(args.test)
    first = json.loads((dev / 'contract.json').read_text(encoding='utf-8'))
    second = json.loads((test / 'contract.json').read_text(encoding='utf-8'))
    selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
    fields = ('smoke_sha256', 'candidate_plan_sha256', 'candidate_results_sha256', 'source_pages_sha256',
              'script_sha256', 'model_revision', 'base_revision', 'weight_sha256', 'max_visual_tokens',
              'dimension', 'device', 'runtime_versions')
    if (first['partition'] != 'dev' or second['partition'] != 'test'
            or second['selection_sha256'] != sha(args.selection)
            or selection.get('status') != 'frozen_before_test_evaluation'
            or sha(selection['development_summary']) != selection['development_summary_sha256']
            or any(first[k] != second[k] for k in fields)
            or sha(Path(args.candidates) / 'results.jsonl') != second['candidate_results_sha256']
            or sha(Path(args.prepared) / 'pages.jsonl') != second['source_pages_sha256']):
        raise ValueError('Page checkpoint contracts are incompatible')
    pages = {x['page_id']: x for x in rows(Path(args.prepared) / 'pages.jsonl')}
    candidates = [x for x in rows(Path(args.candidates) / 'results.jsonl') if x['partition'] == 'test']
    page_ids = list(dict.fromkeys(page for row in candidates for page in row['pages']))
    copied = reused = unavailable = 0
    for page in page_ids:
        source = pages[page]
        expected = {'page_id': page, 'paper_id': source['paper_id'], 'image_sha256': source['image_sha256']}
        if check_marker(test / 'pages', page, expected):
            reused += 1
            continue
        if not check_marker(dev / 'pages', page, expected):
            unavailable += 1
            continue
        marker = json.loads((dev / 'pages' / (page + '.json')).read_text(encoding='utf-8'))
        source_file = dev / 'pages' / marker['vector_file']
        target_file = test / 'pages' / marker['vector_file']
        if target_file.exists():
            if sha(target_file) != marker['vectors_sha256']:
                raise ValueError('Existing copied vector differs; preserved')
        else:
            pending = target_file.with_name(target_file.name + '.' + uuid.uuid4().hex + '.pending')
            with source_file.open('rb') as source_stream, pending.open('xb') as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
            if sha(pending) != marker['vectors_sha256']:
                raise ValueError('Copied vector checksum mismatch; pending file preserved')
            if os.name == 'nt':
                os.rename(pending, target_file)
            else:
                os.link(pending, target_file)
                pending.unlink()
        if sha(target_file) != marker['vectors_sha256']:
            raise ValueError('Copied vector checksum mismatch')
        with (test / 'pages' / (page + '.json')).open('x', encoding='utf-8') as stream:
            json.dump(marker, stream, ensure_ascii=False, sort_keys=True)
        copied += 1
    print(json.dumps({'test_candidate_pages': len(page_ids), 'copied_verified_pages': copied,
                      'already_present': reused, 'not_in_dev': unavailable}))


if __name__ == '__main__':
    main()
