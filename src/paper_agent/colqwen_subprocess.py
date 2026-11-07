"""Run the matching query model in its isolated Python environment."""
import json
import subprocess
from pathlib import Path


class SubprocessColQwenQueryEncoder:
    def __init__(self, python, script, base, adapter, contract, *, device='cuda:0', timeout=180):
        self.python, self.script = str(python), str(script)
        self.base, self.adapter, self.contract = str(base), str(adapter), dict(contract)
        self.device, self.timeout = device, timeout
        for key in ('model_revision', 'base_revision', 'weight_sha256', 'max_visual_tokens'):
            setattr(self, key, self.contract[key])
        if not Path(self.python).is_file() or not Path(self.script).is_file() or timeout <= 0:
            raise ValueError('Invalid query encoder runtime')

    def encode_query(self, query):
        request = {'query': query, 'base': self.base, 'adapter': self.adapter,
                   'contract': self.contract, 'device': self.device}
        result = subprocess.run([self.python, self.script], input=json.dumps(request),
                                text=True, encoding='utf-8', capture_output=True, timeout=self.timeout, check=False)
        if result.returncode:
            raise RuntimeError('Isolated ColQwen query encoder failed')
        response = json.loads(result.stdout)
        if response.get('status') != 'ok':
            raise RuntimeError('Isolated ColQwen query encoder returned an invalid result')
        return response['vectors']
