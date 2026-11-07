"""Local ColQwen query-token encoder for the matching sealed page store."""
import hashlib
from pathlib import Path


class ColQwenQueryEncoder:
    def __init__(self, base, adapter, contract, *, device='cuda:0'):
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        from peft import PeftModel

        self.model_revision = contract['model_revision']
        self.base_revision = contract['base_revision']
        self.weight_sha256 = dict(contract['weight_sha256'])
        self.max_visual_tokens = contract['max_visual_tokens']
        base, adapter = Path(base), Path(adapter)
        for name, expected in contract['weight_sha256'].items():
            path = (adapter if name == 'adapter_model.safetensors' else base) / name
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError('Model weight checksum mismatch: ' + name)
        self.device, self.dimension = device, contract['dimension']
        self.processor = ColQwen2Processor.from_pretrained(str(adapter), local_files_only=True,
            max_num_visual_tokens=contract['max_visual_tokens'])
        if self.processor.query_prefix != 'Query: ':
            raise ValueError('Unexpected processor query prefix')
        model = ColQwen2.from_pretrained(str(base), local_files_only=True, low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32, device_map=device)
        self.model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True).eval()

    def encode_query(self, query):
        import torch
        batch = self.processor.process_queries([query]).to(self.device)
        with torch.inference_mode():
            vectors = self.model(**batch)[0].float().cpu().numpy()
        vectors = vectors[batch['attention_mask'][0].cpu().numpy().astype(bool)]
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension or not len(vectors):
            raise ValueError('Invalid query-token output')
        return vectors

    def encode_page(self, image):
        import torch
        batch = self.processor.process_images([image]).to(self.device)
        with torch.inference_mode():
            vectors = self.model(**batch)[0].float().cpu().numpy()
        active = batch['attention_mask'][0].bool()
        positions = self.processor.get_image_mask(batch)[0][active].nonzero(as_tuple=True)[0].cpu().tolist()
        vectors = vectors[active.cpu().numpy()]
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension or not len(vectors):
            raise ValueError('Invalid page-token output')
        return vectors, {'image_grid_thw': batch['image_grid_thw'].cpu().tolist(),
                         'image_token_positions': positions, 'active_token_count': len(vectors)}
