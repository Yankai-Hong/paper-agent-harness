"""Conservative retrieval plans; subqueries retain the original constraints."""
import re


def explicit_paper_scope(question, catalog):
    """Find unambiguous literal catalog identifiers, not inferred titles or pronouns.

    Exclusions require language understanding; when one appears near an identifier,
    ask for clarification instead of silently expanding or guessing the scope.
    """
    mentions = set()
    negative = False
    for item in catalog:
        identifier = item['paper_id']
        # Tiny IDs such as 'a' occur naturally in English; do not infer from them.
        identifiers = [identifier] if len(identifier) >= 3 or (len(identifier) >= 2 and any(c.isdigit() for c in identifier)) else []
        title = item.get('title','').strip()
        if len(title) >= 6:
            identifiers.append(title)
        for token in identifiers:
            pattern = r'(?<![A-Za-z0-9_-])' + re.escape(token) + r'(?![A-Za-z0-9_-])'
            for match in re.finditer(pattern, question, re.IGNORECASE):
                mentions.add(identifier)
                prefix = question[max(0, match.start()-18):match.start()].lower()
                if re.search(r'(不包括|不包含|排除|除去|除了|except|excluding|exclude|without|not)(?:\s+(?:the\s+)?papers?)?\s*$', prefix):
                    negative = True
    return mentions, negative


def _ordered_mentions(text, catalog):
    """IDs in literal mention order, not a language-model interpretation."""
    positions = []
    for item in catalog:
        pid = item['paper_id']
        tokens = [pid] if len(pid) >= 3 or (len(pid) >= 2 and any(c.isdigit() for c in pid)) else []
        title = item.get('title', '').strip()
        if len(title) >= 6:
            tokens.append(title)
        for token in tokens:
            pattern = r'(?<![A-Za-z0-9_-])' + re.escape(token) + r'(?![A-Za-z0-9_-])'
            positions.extend((match.start(), pid) for match in re.finditer(pattern, text, re.I))
    return list(dict.fromkeys(pid for _, pid in sorted(positions)))


_EXCLUSION = re.compile(r'不包括|不包含|不使用|不要使用|不能使用|不得使用|排除|除去|除了|\bexcept\b|\bexcluding\b|\bexclude\b|\bwithout\b', re.I)
_EXCLUSIVE = re.compile(r'仅限|只限|仅使用|只使用|仅用|只用|仅针对|只针对|只看|仅讨论|只讨论|\bonly\b|\bsolely\b|\bexclusively\b', re.I)


def exclusion_kind(question, catalog):
    """Allow only clearly identified data/time restrictions; uncertain exclusions clarify."""
    matches = list(_EXCLUSION.finditer(question))
    if not matches:
        return None
    for match in matches:
        # Stop at clause boundaries, but leave dots intact (paper IDs contain dots).
        clause = re.split(r'[,，;；。!?！？\n]', question[match.end():], maxsplit=1)[0].strip()
        if not clause or _ordered_mentions(clause, catalog) or re.search(r'论文|文章|文献|研究论文|\bpapers?\b|\barticles?\b', clause, re.I):
            return 'paper_or_uncertain'
        if not re.search(r'数据|样本|年份|年|日期|时间|实验|结果|指标|测量|\bdata\b|\bdatasets?\b|\byears?\b|\bdates?\b|\bexperiments?\b|\bresults?\b', clause, re.I):
            return 'paper_or_uncertain'
    return 'non_paper'


def exclusive_kind(question, catalog):
    """Only classify as a data restriction when its object is explicit."""
    matches=list(_EXCLUSIVE.finditer(question))
    if not matches:
        return None
    for match in matches:
        clause=re.split(r'[,，;；。!?！？\n]',question[match.end():],maxsplit=1)[0].strip()
        if (_ordered_mentions(clause,catalog) or re.search(r'论文|文章|文献|\bpapers?\b|\barticles?\b',clause,re.I) or
                not re.search(r'数据|样本|年份|年|日期|时间|实验|结果|指标|测量|\bdata\b|\bdatasets?\b|\byears?\b|\bdates?\b|\bexperiments?\b|\bresults?\b',clause,re.I)):
            return 'paper_or_uncertain'
    return 'non_paper'


def _reference_text(question, catalog):
    """Mask only clearly local demonstratives attached to an explicit ID/title."""
    result=question
    for item in catalog:
        for token in (item['paper_id'],item.get('title','').strip()):
            if len(token)<3:
                continue
            p=r'(?<![A-Za-z0-9_-])'+re.escape(token)+r'(?![A-Za-z0-9_-])'
            result=re.sub('('+p+r')\s*(?:这篇|该篇)(?:论文|文章|文献)?',r'\1',result,flags=re.I)
            result=re.sub(r'(?:这篇|该篇)(?:论文|文章|文献)?\s*('+p+')',r'\1',result,flags=re.I)
    # In a non-comparison single-paper question, English "its/it" after an
    # explicit paper can refer to that paper or an entity inside it. Do not
    # reinterpret it as a separate historical paper; retain the named scope.
    # Comparisons, temporal anchors and multiple named papers remain ambiguous.
    if (len(_ordered_mentions(result,catalog)) == 1 and
            not re.search(r'比较|对比|刚才|刚刚|上次|上一轮|上轮|之前|此前|先前|\bcompare\w*\b|\bversus\b|\bvs\b|\bcomparison\b|\bdifference\w*\b|\bbetween\b|\bprevious\w*\b|\bearlier\b|\brecent\w*\b',result,re.I)):
        pid=_ordered_mentions(result,catalog)[0]
        item=next(c for c in catalog if c['paper_id']==pid)
        positions=[m.start() for token in (pid,item.get('title','').strip()) if len(token)>=3
                   for m in re.finditer(r'(?<![A-Za-z0-9_-])'+re.escape(token)+r'(?![A-Za-z0-9_-])',result,re.I)]
        if positions:
            first=min(positions)
            result=result[:first]+re.sub(r'\b(?:it|its)\b','',result[first:],flags=re.I)
    return result


def _history_groups(state, catalog):
    """Reconstruct adjacent turn order from retained messages and cited paper IDs."""
    known = {item['paper_id'] for item in catalog}
    groups = []
    for message in state.messages:
        if not isinstance(message, dict):
            continue
        turn = message.get('turn_id')
        role = message.get('role')
        if (not groups or (turn is not None and groups[-1]['turn'] != turn) or
                (turn is None and role == 'user')):
            groups.append({'turn':turn, 'mentioned':[], 'papers':set()})
        group = groups[-1]
        content = message.get('content')
        if role == 'user' and isinstance(content, str):
            names = _ordered_mentions(content, catalog)
            group['mentioned'].extend(p for p in names if p not in group['mentioned'])
            group['papers'].update(names)
        elif role == 'assistant' and isinstance(content, dict):
            for claim in content.get('claims', []):
                for cite in claim.get('citations', []):
                    if cite.get('paper_id') in known:
                        group['papers'].add(cite['paper_id'])
    return groups


def _resolve_reference(question, state, catalog):
    """Return exact prior targets or None for an unresolved/ambiguous reference."""
    groups = _history_groups(state, catalog)
    historical = set().union(*(g['papers'] for g in groups)) if groups else set()
    # Summary paper lists include merely retrieved chunks and entities were
    # inferred by an earlier NLU, so neither is independently sufficient to
    # establish the user's referent or the order of discussion.

    former = bool(re.search(r'前者|后者|後者|\bformer\b|\blatter\b', question, re.I))
    recent = bool(re.search(r'刚才|刚刚|上次|上一轮|上轮|上一回|最近(?:讨论|提到|说到|聊到|一次|那篇|这篇|的论文|的那篇)|\bmost recent\b|\bprevious(?:ly)?\b|\blast (?:paper|one|turn|discuss)', question, re.I))
    plural = bool(re.search(r'它们|这两篇|那两篇|\bthem\b|\bthose\b', question, re.I))
    if former:
        # Only the user's explicit prior ordering establishes former/latter.
        if not groups or len(groups[-1]['mentioned']) != 2 or groups[-1]['papers'] != set(groups[-1]['mentioned']):
            return None
        order = groups[-1]['mentioned']
        return {order[1] if re.search(r'后者|後者|\blatter\b', question, re.I) else order[0]}
    if recent:
        if groups:
            return groups[-1]['papers'] if len(groups[-1]['papers']) == 1 else None
        return None
    if plural:
        return groups[-1]['papers'] if groups and len(groups[-1]['papers']) == 2 else None
    # Generic singular pronouns do not necessarily mean the last paper. Only a
    # unique session-wide candidate is safe; all others require clarification.
    # An archive means the retained window is incomplete. Summary/entity data
    # may veto a mistaken uniqueness claim, but may not establish the target.
    if state.archive_ids or state.memory.summary.get('archive_id'):
        return None
    known = {item['paper_id'] for item in catalog}
    hints = {p for key in ('discussed_papers','paper_order')
             for p in state.memory.summary.get(key,[]) if p in known}
    hints.update(v for v in state.memory.entities.values() if isinstance(v,str) and v in known)
    if hints - historical:
        return None
    return historical if len(historical) == 1 else None


def scope_guard(question, catalog, nlu_papers, state, standalone_query=None, resolved_entities=None):
    """Conservatively reconcile explicit mentions, exclusions and prior context.

    A mention does not mean an exclusive scope. Additional IDs must have a
    literal or structured origin in the current session, not merely be known
    to the catalog. The parser intentionally refuses uncertain exclusions.
    """
    mentioned, negative = explicit_paper_scope(question, catalog)
    exclusive = exclusive_kind(question,catalog) == 'paper_or_uncertain'
    exclusion = exclusion_kind(question, catalog)
    if negative or exclusion == 'paper_or_uncertain':
        return 'explicit_targets_unresolved'
    chosen = None if nlu_papers is None else set(nlu_papers)
    ref_text=_reference_text(question,catalog)
    has_reference = bool(re.search(r'刚才|刚刚|之前|以前|上次|上一轮|上轮|上一回|此前|先前|前面|最近(?:讨论|提到|说到|聊到|一次|那篇|这篇|的论文|的那篇)|那篇|这篇|它|前者|后者|後者|这两篇|那两篇|\bearlier\b|\bprevious\w*\b|\brecent\w*\b|\bthat paper\b|\bthe other\b|\bit\b|\bits\b|\bthem\b|\bthose\b|\bformer\b|\blatter\b',ref_text,re.I))
    # Scope and rewrite are one contract: otherwise a legal citation from B can
    # still answer a rewritten comparison about the wrong paper C.
    rewritten=set(_ordered_mentions(standalone_query,catalog)) if isinstance(standalone_query,str) else set()
    if rewritten and (chosen is None or rewritten != chosen):
        return 'explicit_targets_unresolved'
    for key,value in (resolved_entities or {}).items():
        if not isinstance(value,str):
            continue
        referenced=set(_ordered_mentions(value,catalog))
        if referenced and (chosen is None or not referenced <= chosen):
            return 'explicit_targets_unresolved'
        if has_reference and re.search(r'它|刚才|刚刚|上一轮|上轮|那篇|这篇|前者|后者|後者|\bit\b|\bits\b|\bformer\b|\blatter\b|\bthat\b|\brecent\b',str(key),re.I):
            target=_resolve_reference(ref_text,state,catalog)
            if not referenced or target is None or referenced != target:
                return 'explicit_targets_unresolved'
    if exclusive:
        # A named ID plus an unresolved historical reference cannot be reduced
        # to the named ID; that would silently drop an explicitly requested target.
        if has_reference or not mentioned or chosen != mentioned:
            return 'explicit_targets_unresolved'
        return None
    if not mentioned and not has_reference:
        return None
    if chosen is None or (has_reference and not chosen) or not mentioned <= chosen:
        return 'explicit_targets_unresolved'
    extra = chosen - mentioned
    if not has_reference:
        return None if not extra else 'explicit_targets_unresolved'
    resolved = _resolve_reference(ref_text, state, catalog)
    if resolved is None or chosen != mentioned | resolved:
        return 'explicit_targets_unresolved'
    # A comparison with a historical singular target cannot silently collapse
    # both sides onto the same explicitly named paper.
    if mentioned and resolved <= mentioned and re.search(r'比较|对比|比较一下|\bcompare\b|\bversus\b|\bvs\b',question,re.I):
        return 'explicit_targets_unresolved'
    return None


def plan(question, nlu, enabled=False):
    base = nlu['standalone_query']
    papers = nlu.get('paper_ids')
    single = [{'id':'main','query':base,'paper_ids':papers,'target':question}]
    if not enabled:
        return single
    complex_question = (len(papers or []) > 1 or nlu.get('intent') == 'compare' or
                        bool(re.search(r'分别|以及|优缺点|多个|多维|各方面|\band\b|\bversus\b',question,re.I)))
    if not complex_question:
        return single
    proposed = nlu.get('sub_questions') or []
    if not proposed and 1 < len(papers or []) <= 3:
        proposed = [{'query':base,'paper_ids':[p],'target':p} for p in papers]
    if not isinstance(proposed, list) or not 2 <= len(proposed) <= 3:
        return single
    selected, seen = [], set()
    for item in proposed:
        if not isinstance(item,dict):
            return single
        scope = item.get('paper_ids')
        query = item.get('query')
        target = item.get('target')
        # No unbounded expansion; even a valid catalog ID cannot escape parent scope.
        if not isinstance(query,str) or not query.strip() or len(query)>2000 or not isinstance(target,(str,type(None))):
            return single
        query=query.strip()
        if papers is None:
            # A null parent permits corpus-wide subqueries; a subset cannot be
            # trusted to cover the unspecified scope without an independent plan.
            if scope is not None:
                return single
        elif not isinstance(scope,list) or not scope or any(not isinstance(p,str) for p in scope) or not set(scope)<=set(papers):
            return single
        signature = (query.casefold(),tuple(sorted(scope)) if scope is not None else None)
        if signature in seen:
            continue
        seen.add(signature)
        selected.append({'id':f'g{len(selected)+1}', 'query':question+'\n'+base+'\n'+query,
                         'paper_ids':scope,'target':(target or query)[:300]})
    if len(selected)<2 or (papers is not None and set().union(*(set(s['paper_ids']) for s in selected)) != set(papers)):
        return single
    return selected
