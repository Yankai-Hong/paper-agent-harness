import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path('scripts').resolve()))
from colqwen_encode_candidates import check_marker, save_vector


def test_page_checkpoint_is_resumable_and_checks_source_and_content(tmp_path):
    expected = {'page_id': '1_1', 'image_sha256': 'a' * 64}
    assert not check_marker(tmp_path, '1_1', expected)
    # Orphan files from interrupted writes do not constitute a checkpoint.
    (tmp_path / '1_1-interrupted.npz').write_bytes(b'partial')
    assert not check_marker(tmp_path, '1_1', expected)
    record = save_vector(tmp_path, '1_1', np.eye(2, dtype=np.float32), expected)
    assert check_marker(tmp_path, '1_1', expected)
    with pytest.raises(ValueError, match='metadata mismatch'):
        check_marker(tmp_path, '1_1', {'page_id': '1_1', 'image_sha256': 'b' * 64})
    (tmp_path / record['vector_file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='content mismatch'):
        check_marker(tmp_path, '1_1', expected)


def test_invalid_or_unpadded_vectors_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='invalid token'):
        save_vector(tmp_path, '1_1', [[0., 0.]], {'page_id': '1_1'})
