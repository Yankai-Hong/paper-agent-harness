from paper_agent.pipeline import _answer_language, _corpus_language, PaperAgentState
from test_revision_decomposition import make_complex_pipeline


EN = ('The proposed method uses the context of each entity and the knowledge base to predict its type. '
      'We evaluate our approach on the datasets and report the results for each experiment. ') * 3
ZH = '本文介绍使用知识库和上下文信息对实体进行分类的方法，并在多个数据集上报告实验结果。' * 5


def test_corpus_language_is_scoped_and_conservative():
    items = [{'paper_id':'en', 'text':EN}, {'paper_id':'zh', 'text':ZH}]
    assert _corpus_language(items, 'en') == 'en'
    assert _corpus_language(items, 'zh') == 'zh'
    assert _corpus_language(items) == 'unknown'
    assert _corpus_language([{'paper_id':'fr', 'text':'Les résultats de cette expérience montrent une amélioration sur plusieurs jeux de données. ' * 4}]) == 'unknown'
    assert _corpus_language([{'paper_id':'x', 'text':'Too short.'}]) == 'unknown'


def test_answer_language_ignores_english_scope_prefix():
    assert _answer_language('Only paper 1909.12079: 这篇论文使用什么知识库？') == 'zh'
    assert _answer_language('Only paper 1909.12079: What knowledge base is used?') == 'en'
    assert _answer_language('Only paper 1909.12079: ?') == 'unknown'


def test_nlu_receives_language_without_losing_original_question(tmp_path):
    nlu = {'intent':'fact', 'standalone_query':'Which knowledge base is used?', 'paper_ids':['paperA'],
           'resolved_entities':{}, 'needs_clarification':False}
    pipeline, _, calls = make_complex_pipeline(tmp_path, nlu=nlu, decomposition=False)
    for item in pipeline.retriever.by_id.values():
        item['text'] = EN
    question = 'Only paper paperA: 这篇论文使用什么知识库？'
    assert pipeline.run(question, PaperAgentState('s'), 'language')['answer']['decision'] == 'answer'
    nlu_payload = next(payload for stage,payload in calls if stage == 'nlu')
    assert nlu_payload['retrieval_corpus_language'] == 'en'
    assert nlu_payload['question'] == question
    for stage in ('reflect','generate'):
        assert next(payload for name,payload in calls if name == stage)['question'] == question
