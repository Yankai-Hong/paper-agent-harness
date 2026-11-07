"""Bounded batched MaxSim over validated token matrices, with no averaging."""
import numpy as np

from .colqwen_store import _matrix


def score_maxsim_batch(query, documents, *, device='cpu', max_total_tokens=100000):
    import torch

    q = _matrix(query)
    if not documents or len(documents) > 1000 or type(max_total_tokens) is not int or max_total_tokens < 1:
        raise ValueError('Invalid candidate batch bounds')
    matrices = [_matrix(doc, q.shape[1]) for doc in documents]
    lengths = [len(doc) for doc in matrices]
    if sum(lengths) > max_total_tokens:
        raise ValueError('Candidate token batch exceeds the configured limit')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA scoring device is unavailable')
    # Never use reduced-precision TF32 for a supposedly float32 comparison.
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
    all_tokens = torch.from_numpy(np.concatenate(matrices, axis=0)).to(device=device, dtype=torch.float32)
    query_tensor = torch.from_numpy(q).to(device=device, dtype=torch.float32)
    owners = torch.repeat_interleave(torch.arange(len(matrices), device=device),
                                     torch.tensor(lengths, dtype=torch.long, device=device))
    similarities = query_tensor @ all_tokens.T
    maxima = torch.full((len(q), len(matrices)), float('-inf'), device=device, dtype=torch.float32)
    maxima.scatter_reduce_(1, owners.unsqueeze(0).expand(len(q), -1), similarities, reduce='amax', include_self=True)
    values = maxima.sum(dim=0).detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError('Invalid MaxSim scores')
    return [float(value) for value in values]
