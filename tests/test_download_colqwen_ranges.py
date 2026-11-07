import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('download_colqwen_ranges', Path('scripts/download_colqwen_ranges.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_range_is_validated_before_write_and_resume_verifies_content(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'SHARDS', [('a.safetensors', 4, 'unused')])

    class Response:
        status_code = 206
        headers = {'Content-Range': 'bytes 0-3/4'}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def iter_content(self, _): yield b'abcd'

    monkeypatch.setattr(module.requests, 'get', lambda *a, **kw: Response())
    first = module.fetch(tmp_path, 0, 0, 0, 3)
    assert first.read_bytes() == b'abcd'
    assert module.ready(tmp_path, 0, 0, 0, 3) == first
    first.write_bytes(b'xxxx')
    assert module.ready(tmp_path, 0, 0, 0, 3) is None

    Response.headers = {'Content-Range': 'bytes 1-3/4'}
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match='unexpected range'):
        module.fetch(tmp_path, 0, 0, 0, 3)
    assert set(tmp_path.iterdir()) == before
