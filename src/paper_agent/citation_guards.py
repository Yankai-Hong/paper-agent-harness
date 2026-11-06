"""Conservative per-claim numeric coverage checks for cited text.

This guard detects missing explicit numeric conditions; it does not prove
semantic entailment or examine numbers in images. A cited visual region makes
the mechanical result unknown instead of accepting or rejecting its contents.
"""
from decimal import Decimal, InvalidOperation
import re

_PATTERN = re.compile(
    r'(?<![A-Za-z0-9_.])(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)'
    r'(?:[ \t]*(?P<unit>%|％|percent(?:age)?|milliseconds?|msecs?|ms|seconds?|secs?|hours?|hrs?|'
    r'years?|days?|million|billion|thousand|小时|毫秒|秒|年份|年|天|万|千|百万|亿)(?![A-Za-z]))?',
    re.IGNORECASE)

_UNITS = {
    '%':'percent', '％':'percent', 'percent':'percent', 'percentage':'percent',
    'second':'second','seconds':'second','sec':'second','secs':'second','秒':'second',
    'millisecond':'millisecond','milliseconds':'millisecond','msec':'millisecond','msecs':'millisecond','ms':'millisecond','毫秒':'millisecond',
    'hour':'hour','hours':'hour','hr':'hour','hrs':'hour','小时':'hour',
    'year':'year','years':'year','年':'year','年份':'year',
    'day':'day','days':'day','天':'day',
}
_SCALES = {'thousand':1000, '千':1000, '万':10000, 'million':1000000,
           '百万':1000000, 'billion':1000000000, '亿':100000000}

_DEPENDENT = re.compile(r'\bthis\s+(?:is|was|effort|result|value|score|time|comparison|approach|method)\b|'
                        r'\b(?:it|they|these|those)\s+(?:is|are|was|were|has|have)\b|'
                        r'这一结果|该结果|上述结果|前者|后者', re.I)
_TEST_CONDITION = re.compile(r'\btest(?:ing)?\s*(?:[- ]?set|split|data|perplexity|accuracy|error|score)\b|测试集|测试数据|测试误差|测试得分', re.I)
_TEST_EVIDENCE = re.compile(r'\btest(?:ing)?\b|测试', re.I)


def check_claim_conditions(claims, evidence):
    """Reject obvious dangling claim references and unsupported test qualifiers.

    These checks detect narrow problems only; passing does not prove entailment.
    Visual regions cannot mechanically establish a dataset-split qualifier.
    """
    by_key = {(row.get('paper_id'),row.get('chunk_id')):row for row in evidence}
    out = []
    for claim in claims:
        text = claim.get('text','')
        refs = [by_key.get((r.get('paper_id'),r.get('chunk_id'))) for r in claim.get('citations',[])]
        status = 'ok'
        if _DEPENDENT.search(text):
            status = 'context_dependent'
        elif _TEST_CONDITION.search(text) and not any(
                row and row.get('evidence_type') not in {'figure','table'} and
                _TEST_EVIDENCE.search(row.get('text','')) for row in refs):
            status = 'test_condition_unknown'
        out.append({'claim_key':claim.get('key'),'status':status})
    return out


def _numbers(text):
    values = []
    for match in _PATTERN.finditer(text or ''):
        try:
            number = Decimal(match.group('number').replace(',', ''))
        except InvalidOperation:
            continue
        unit = (match.group('unit') or '').lower()
        if unit in _SCALES:
            number *= _SCALES[unit]
            unit = ''
        else:
            unit = _UNITS.get(unit, '')
        if not unit and text[max(0, match.start()-3):match.start()] == '百分之':
            unit = 'percent'
        values.append((number.normalize(), unit))
    return values


def check_claim_numbers(claims, evidence):
    """Return per-key status without treating numeric overlap as entailment."""
    by_key = {(row.get('paper_id'), row.get('chunk_id')):row for row in evidence}
    results = []
    for claim in claims:
        refs = [by_key.get((r.get('paper_id'),r.get('chunk_id'))) for r in claim.get('citations', [])]
        # A literal, cited corpus identifier is an identity label, not a
        # reported measurement or year inside the cited PDF region. Mask only
        # the complete identifier; standalone numeric facts remain checked.
        text = claim.get('text', '')
        for pid in {r.get('paper_id') for r in claim.get('citations', []) if isinstance(r.get('paper_id'), str)}:
            if re.search(r'[A-Za-z.\-]', pid):
                text = re.sub(r'(?<![A-Za-z0-9_.-])' + re.escape(pid) + r'(?![A-Za-z0-9_.-])',
                              lambda m:' ' * len(m.group(0)), text, flags=re.I)
        numbers = _numbers(text)
        if any(row is None for row in refs):
            results.append({'claim_key':claim.get('key'), 'status':'invalid_reference'})
        elif not numbers:
            results.append({'claim_key':claim.get('key'), 'status':'no_explicit_number'})
        elif any(row.get('evidence_type') in {'figure','table'} for row in refs):
            results.append({'claim_key':claim.get('key'), 'status':'unknown_visual'})
        else:
            found = [value for row in refs for value in _numbers(row.get('text', ''))]
            missing = [(n,u) for n,u in numbers if not any(n == x and (not u or u == v) for x,v in found)]
            results.append({'claim_key':claim.get('key'), 'status':'missing' if missing else 'covered',
                            'missing_count':len(missing),
                            'missing_values':[{'value':str(n), 'unit':u} for n,u in missing]})
    return results


def check_question_collection_conditions(question, claims):
    """Conservatively retain explicit collection sizes in numerical claims."""
    matches = list(re.finditer(r'(?i)\b(?:among|within|from\s+a\s+collection\s+of|in\s+a\s+collection\s+of)\s+'
                               r'(\d[\d,]*)\s+([a-z]+)\b', question))
    if len(matches) != 1:
        return []
    count = matches[0].group(1).replace(',', '')
    noun = matches[0].group(2).lower()
    results = []
    for claim in claims:
        text = claim.get('text', '')
        if not _numbers(text):
            continue
        found = any(m.group(1).replace(',', '') == count and m.group(2).lower() == noun
                    for m in re.finditer(r'(?i)\b(\d[\d,]*)\s+([a-z]+)\b', text))
        results.append({'claim_key':claim.get('key'), 'status':'ok' if found else 'collection_condition_missing'})
    return results
