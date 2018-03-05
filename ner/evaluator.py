"""Evaluate a NER/bioconcept tagger
"""

import re
import numpy as np
import pandas as pd
from tqdm import tqdm

def _entities_from_tagger(tagger, doc_id, doc_text):
    return [e['entity'].lower() for e in tagger(doc_id, doc_text)]

def evaluate(tagger, data, multiple=False, snippets=False):
    list_type = data.get_questions_of_type('list')
    factoid_type = data.get_questions_of_type('factoid')
    results = []

    missed = 0
    for question in tqdm(list_type + factoid_type):
        answers = [answer.lower() for answer in _flatten(question.exact_answer_ref)]
        docs = question.snippets if snippets else question.documents
        try:
            if multiple:
                doc_ids = [document.doc_id for document in docs]
                doc_texts = [document.text for document in docs]
                entities = _entities_from_tagger(tagger, doc_ids, doc_texts)
            else:
                entities = []
                for document in docs:
                    entities += _entities_from_tagger(tagger, document.doc_id, document.text)
        except:
            missed += 1
            continue

        entities = list(np.unique(entities))
        exact_matches = sum([answer in entities for answer in answers])
        soft_matches = _soft_matches(answers, entities)
        results.append({
            'qid': question.qid,
            'answers': answers,
            'entities': entities,
            'exact_matches': exact_matches,
            'total_answers': len(answers),
            'total_entities': len(entities),
            'soft_matches': soft_matches,
            'type': question.type
        })

    return pd.DataFrame(results), missed

def summary_from_results(results, data):
    list_type = results[results.type == 'list']
    factoid_type = results[results.type == 'factoid']
    v_size = pd.DataFrame([{'qid': i.qid, 'V': i.V_snippets} for i in data.questions])
    rows = []

    for df, qtype in [(list_type, 'list'), (factoid_type, 'factoid')]:
        N = len(df) * 1.0
        exact_matched_questions = len(df[df.exact_matches > 0]) / N
        soft_match_questions = len(df[df.soft_matches > 0]) / N

        joined = df.join(v_size.set_index('qid'), on=['qid'], lsuffix='', rsuffix='_r')
        joined = joined[joined.V > 0]
        frac = (joined.total_entities / joined.V).mean()

        n_exact_answers = df.total_answers.sum() * 1.0
        exact_matched_answers = df.exact_matches.sum() / n_exact_answers
        soft_match_answers = df.soft_matches.sum() / n_exact_answers

        rows.append({
            'type': qtype,
            'exact_matched_questions': exact_matched_questions,
            'soft_match_questions': soft_match_questions,
            'exact_matched_answers': exact_matched_answers,
            'soft_match_answers': soft_match_answers,
            'n_entities/total_tokens': frac,
        })

    return pd.DataFrame(rows, columns=['type', 'exact_matched_questions', 'soft_match_questions', 'exact_matched_answers', 'soft_match_answers', 'n_entities/total_tokens'])

def clean_entities(df, data):
    rows = []
    for itr, row in df.iterrows():
        entities = list(np.unique(_clean_tokens(row.entities)))
        answers = row.answers
        exact_matches = sum([answer in entities for answer in answers])
        soft_matches = _soft_matches(answers, entities)
        rows.append({
            'qid': row.qid,
            'answers': answers,
            'entities': entities,
            'exact_matches': exact_matches,
            'total_answers': len(answers),
            'total_entities': len(entities),
            'soft_matches': soft_matches,
            'type': row.type
        })
    
    return pd.DataFrame(rows)

def _clean_tokens(tokens):
    return [i for i in tokens if len(re.sub('\\W+', '', i)) > 0]

def ensemble_tags(results, typ='union'):
    df = results[0]
    if len(results) == 1:
        return df

    entities = {}
    answers = {}
    types = {}
    first = True
    for df2 in results[1:]:
        for itr, row in df.join(df2.set_index('qid'), on=['qid'], lsuffix='_l', rsuffix='_r', how='inner').iterrows():
            if row.qid not in entities:
                entities[row.qid] = []
            if first:
                entities[row.qid].append(row.entities_l)
            entities[row.qid].append(row.entities_r)
            if row.qid not in answers:
                answers[row.qid] = row.answers_l
                types[row.qid] = row.type_l
        first = False
    
    rows = []
    for qid, answer in answers.items():
        rows.append({
            'qid': qid,
            'answers': answer,
            'type': types[qid],
            'entities': _combine(entities[qid], typ=typ)            
        })

    return pd.DataFrame(rows)

def _combine(arr, typ='union'):
    sets = [set(i) for i in arr]
    initial = sets[0]
    for s in sets[1:]:
        initial = initial.union(s) if typ == 'union' else initial.intersection(s)
    return list(initial)


def _soft_matches(answers, entities):
    answers = list(map(_normalize, answers))
    entities = list(map(_normalize, entities))
    matches = 0
    for answer in answers:
        for entity in entities:
            if answer in entity or entity in answer:
                matches += 1
                break
    return matches
    
def _normalize(string):
    return re.sub(r'\W+', '', string).lower()

def _flatten(l):
    if not isinstance(l, list):
        return [l]
    flattened = []
    for item in l:
        flattened += _flatten(item)
    return flattened
