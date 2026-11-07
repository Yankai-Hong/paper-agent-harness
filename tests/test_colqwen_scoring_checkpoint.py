import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path('scripts').resolve()))
from colqwen_candidate_eval import load_scoring_checkpoint, save_scoring_checkpoint


def test_scoring_checkpoints_are_immutable_and_detect_changes(tmp_path):
    path = tmp_path / 'case.json'
    record = {'case_id': 'dev-1', 'candidate_rank': 2, 'top20': ['p1', 'p2']}
    assert load_scoring_checkpoint(path, 'contract', 'query', 'pages', 'dev-1') is None
    save_scoring_checkpoint(path, 'contract', 'query', 'pages', 1e-6, record)
    assert load_scoring_checkpoint(path, 'contract', 'query', 'pages', 'dev-1')['result'] == record
    with pytest.raises(FileExistsError):
        save_scoring_checkpoint(path, 'contract', 'query', 'pages', 1e-6, record)
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        load_scoring_checkpoint(path, 'contract', 'changed-query', 'pages', 'dev-1')
    data = json.loads(path.read_text(encoding='utf-8'))
    data['result']['candidate_rank'] = 1
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        load_scoring_checkpoint(path, 'contract', 'query', 'pages', 'dev-1')
