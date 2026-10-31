"""Exercise the fixed local CLIP pair in separate model processes.

This is an encoding smoke test, not a retrieval or QA accuracy measurement.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from paper_agent.visual_retrieval import SubprocessEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-python', required=True)
    parser.add_argument('--image-model', required=True)
    parser.add_argument('--text-model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-fallback', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite an existing result')
    versions = subprocess.run([args.visual_python, '-c',
        "import json,sys,importlib.metadata as m; import torch; print(json.dumps({'python':sys.version,'torch':torch.__version__,'sentence_transformers':m.version('sentence-transformers'),'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda}))"],
        capture_output=True, text=True, check=True, timeout=60)
    result = {'versions': json.loads(versions.stdout.strip().splitlines()[-1]), 'attempts': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    devices = [args.device] + (['cpu'] if args.cpu_fallback and args.device != 'cpu' else [])
    for device in devices:
        attempt = {'device': device, 'status': 'error'}
        result['attempts'].append(attempt)
        started = time.perf_counter()
        try:
            encoder = SubprocessEncoder(args.visual_python, args.image_model, args.text_model, device=device)
            result['model_provenance'] = encoder.provenance
            start = time.perf_counter()
            image = encoder.encode_images([args.image])
            attempt['image_elapsed_seconds'] = time.perf_counter() - start
            start = time.perf_counter()
            text = encoder.encode_texts(['A neural network architecture diagram', '神经网络架构图'])
            attempt['text_elapsed_seconds'] = time.perf_counter() - start
            if image.shape != (1, 512) or text.shape != (2, 512):
                raise ValueError('Aligned model embeddings must be 512-dimensional')
            image_norm = np.linalg.norm(image, axis=1)
            text_norm = np.linalg.norm(text, axis=1)
            if not np.allclose(image_norm, 1, atol=1e-5) or not np.allclose(text_norm, 1, atol=1e-5):
                raise ValueError('Vectors are not normalized')
            similarities = (text @ image.T).reshape(-1)
            if not all(math.isfinite(float(s)) for s in similarities):
                raise ValueError('Non-finite similarity')
            attempt.update(status='ok', image_shape=list(image.shape), text_shape=list(text.shape),
                           image_norms=image_norm.tolist(), text_norms=text_norm.tolist(),
                           cosine_english=float(similarities[0]), cosine_chinese=float(similarities[1]))
        except Exception as exc:
            attempt['error_type'] = type(exc).__name__
        finally:
            attempt['elapsed_seconds'] = time.perf_counter() - started
        if attempt['status'] == 'ok':
            break
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not any(a['status'] == 'ok' for a in result['attempts']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
