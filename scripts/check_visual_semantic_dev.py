"""Run a real BGE visual-caption retrieval smoke test on development data."""
import json
from pathlib import Path
from paper_agent.retrieval import BGEEncoder
from paper_agent.visual_retrieval import SubprocessEncoder, VisualRetriever
from paper_agent.visual_text_retrieval import VisualTextRetriever

encoder = BGEEncoder('models\\bge-m3', 'cpu', 4, 768)
visual_encoder = SubprocessEncoder('.venv-mm-20261030/Scripts/python.exe',
    'models/multimodal-v1/clip-image-v1', 'models/multimodal-v1/clip-text-v1', 'cpu')
visual = VisualRetriever('indexes/multimodal-v5-agent-reviewed',
    'data/raw/multimodal-eval/docling-v5-agent-reviewed/crops', visual_encoder)
semantic = VisualTextRetriever('indexes/multimodal-v5-bge-visual-text-dev-20261031', visual, encoder)
queries = [('attention heatmaps for teacher and student models', 'caption'),
           ('entity typing benchmark results', 'table')]
result = {kind:semantic.search(query, kind=kind, top_k=5) for query,kind in queries}
output = Path('runs/multimodal-dev-semantic-20261031.json')
with output.open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print({kind:len(result[kind]['hits']) for _,kind in queries})
