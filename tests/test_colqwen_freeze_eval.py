import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path('scripts').resolve()))
from colqwen_smoke import sha

spec = importlib.util.spec_from_file_location('colqwen_freeze_eval', Path('scripts/colqwen_freeze_eval.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_test_configuration_freeze_checks_dev_artifact_hashes(tmp_path, monkeypatch):
    results = tmp_path / 'results.jsonl'
    results.write_text('{}\n', encoding='utf-8')
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'partition': 'dev', 'candidate_plan_sha256': 'p', 'candidate_results_sha256': 'r',
        'model_revision': 'm', 'base_revision': 'b', 'weight_sha256': {}, 'max_visual_tokens': 256, 'dimension': 128,
        'device': 'cuda:0', 'runtime_versions': {'torch': 'example'},
        'script_sha256': sha('scripts/colqwen_encode_candidates.py')}), encoding='utf-8')
    summary = tmp_path / 'results.jsonl.summary.json'
    summary.write_text(json.dumps({'status': 'complete', 'partition': 'dev', 'metrics': {'standard': {}, 'visual': {}},
        'results_sha256': sha(results), 'encoding_contract_sha256': sha(contract),
        'evaluation_script_sha256': sha('scripts/colqwen_candidate_eval.py'),
        'store_code_sha256': sha('src/paper_agent/resumable_colqwen.py'),
        'maxsim_code_sha256': sha('src/paper_agent/colqwen_store.py'),
        'maxsim_batch_code_sha256': sha('src/paper_agent/maxsim_batch.py'),
        'score_device': 'cuda:0', 'cache_pages': 256, 'scoring_torch_version': 'example'}), encoding='utf-8')
    output = tmp_path / 'selection.json'
    monkeypatch.setattr(sys, 'argv', ['freeze', '--development-summary', str(summary), '--encoding-contract', str(contract), '--output', str(output)])
    module.main()
    assert json.loads(output.read_text(encoding='utf-8'))['status'] == 'frozen_before_test_evaluation'
    results.write_text('changed\n', encoding='utf-8')
    with pytest.raises(ValueError, match='results or encoding'):
        module.main()
