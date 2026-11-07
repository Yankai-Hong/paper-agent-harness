import numpy as np
import pytest

from paper_agent.colqwen_store import maxsim
from paper_agent.maxsim_batch import score_maxsim_batch


def test_batch_matches_unpadded_per_page_maxsim():
    rng = np.random.default_rng(8)
    def vectors(n):
        value = rng.standard_normal((n, 16)).astype(np.float32)
        return value / np.linalg.norm(value, axis=1, keepdims=True)
    query = vectors(7)
    docs = [vectors(1), vectors(4), vectors(9)]
    assert score_maxsim_batch(query, docs) == pytest.approx([maxsim(query, d) for d in docs], abs=1e-5)
    assert score_maxsim_batch([[1., 0.]], [[[-1., 0.]]]) == pytest.approx([-1.])
    with pytest.raises(ValueError, match='token batch'):
        score_maxsim_batch(query, docs, max_total_tokens=2)
