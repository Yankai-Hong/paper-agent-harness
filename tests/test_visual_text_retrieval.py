import hashlib

import numpy as np
import pytest

from paper_agent.storage import digest, write_jsonl
from paper_agent.visual_text_retrieval import VisualTextRetriever, build_visual_text_index


class BGEFixture:
    model_id = 'fixture-bge'
    max_length = 768
    def encode(self, texts):
        return np.asarray([[1., 0.] if 'cat' in text else [0., 1.] for text in texts], dtype=np.float32)


def make_item(root, name, paper, number, kind='figure'):
    (root/name).write_bytes(name.encode())
    return {'paper_id':paper,'chunk_id':name,'evidence_type':kind,'figure_number':number,
            'text':'unverified full text', 'caption':f'{number}: cat caption',
            'page':1,'bbox':[1,2,20,30],'page_size':[100,100],
            'coordinate_system':'top-left-points','page_rotation':0,
            'image_path':name,'image_sha256':hashlib.sha256(name.encode()).hexdigest(),
            **({'table_text':'cells dog data','table_text_status':'docling_extracted_unverified'} if kind == 'table' else {})}


def test_bge_visual_text_preserves_entity_scope_and_version(tmp_path):
    root = tmp_path/'images'; root.mkdir()
    items = [make_item(root,'one.png','p1','Figure 1'), make_item(root,'two.png','p2','Table 2','table')]
    src=tmp_path/'evidence.jsonl'; write_jsonl(src,items)
    dest=tmp_path/'semantic'; meta=build_visual_text_index(src,dest,root,BGEFixture())
    assert meta['dimension']==2 and meta['representations']=={'caption':2,'table':1}
    class Visual: pass
    visual=Visual();visual.evidence=items;visual.metadata={'corpus_hash':digest(items)};visual.image_root=root
    retrieval=VisualTextRetriever(dest,visual,BGEFixture())
    assert retrieval.search('cat', paper_ids=['p2'], figure_number='Table 2')['hits'][0]['chunk_id']=='two.png'
    assert retrieval.search('dog', kind='table')['hits'][0]['chunk_id']=='two.png'
    assert retrieval.search('dog', kind='table', paper_ids=['p1'])['hits']==[]
    assert retrieval.search('cat', figure_number='Figure 9')['hits']==[]
    with pytest.raises(ValueError, match='overwrite'):
        build_visual_text_index(src,dest,root,BGEFixture())
    visual.metadata={'corpus_hash':'mismatch'}
    with pytest.raises(ValueError, match='does not match'):
        VisualTextRetriever(dest,visual,BGEFixture())
