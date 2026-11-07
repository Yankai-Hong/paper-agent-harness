import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reuse_only_compatible_verified_page_checkpoints(tmp_path):
    dev, test, candidates, prepared = [tmp_path / x for x in ('dev', 'test', 'candidates', 'prepared')]
    for folder in (dev / 'pages', test / 'pages', candidates, prepared):
        folder.mkdir(parents=True)
    pages = prepared / 'pages.jsonl'
    pages.write_text(json.dumps({'page_id': '1', 'paper_id': 'p', 'image_sha256': 'image'}) + '\n')
    results = candidates / 'results.jsonl'
    results.write_text(json.dumps({'partition': 'test', 'pages': ['1']}) + '\n')
    summary = tmp_path / 'summary.json'
    summary.write_text('{}')
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'status': 'frozen_before_test_evaluation', 'development_summary': str(summary),
                                     'development_summary_sha256': digest(summary)}))
    shared = dict(smoke_sha256='smoke', candidate_plan_sha256='plan', candidate_results_sha256=digest(results),
                  source_pages_sha256=digest(pages), script_sha256='script', model_revision='model',
                  base_revision='base', weight_sha256={}, max_visual_tokens=256, dimension=2,
                  device='cpu', runtime_versions={})
    (dev / 'contract.json').write_text(json.dumps({**shared, 'partition': 'dev'}))
    (test / 'contract.json').write_text(json.dumps({**shared, 'partition': 'test', 'selection_sha256': digest(selection)}))
    vector = dev / 'pages' / '1-vector.npz'
    vector.write_bytes(b'verified vector')
    (dev / 'pages' / '1.json').write_text(json.dumps({'page_id': '1', 'paper_id': 'p', 'image_sha256': 'image',
                                                       'vector_file': vector.name, 'vectors_sha256': digest(vector)}))
    script = Path(__file__).parents[1] / 'scripts' / 'colqwen_reuse_pages.py'
    command = [sys.executable, str(script), '--dev', str(dev), '--test', str(test), '--selection', str(selection),
               '--candidates', str(candidates), '--prepared', str(prepared)]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout)['copied_verified_pages'] == 1
    assert json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)['already_present'] == 1
    vector.write_bytes(b'changed')
    (test / 'pages' / '1.json').unlink()
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
