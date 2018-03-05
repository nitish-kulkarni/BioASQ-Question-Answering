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

def summary_from_results(results):
    list_type = results[results.type == 'list']
    factoid_type = results[results.type == 'factoid']
    rows = []

    for df, qtype in [(list_type, 'list'), (factoid_type, 'factoid')]:
        N = len(df) * 1.0
        exact_matched_questions = len(df[df.exact_matches > 0]) / N
        soft_match_questions = len(df[df.soft_matches > 0]) / N

        n_exact_answers = df.total_answers.sum() * 1.0
        exact_matched_answers = df.exact_matches.sum() / n_exact_answers
        soft_match_answers = df.soft_matches.sum() / n_exact_answers

        rows.append({
            'type': qtype,
            'exact_matched_questions': exact_matched_questions,
            'soft_match_questions': soft_match_questions,
            'exact_matched_answers': exact_matched_answers,
            'soft_match_answers': soft_match_answers,
        })

    return pd.DataFrame(rows, columns=['type', 'exact_matched_questions', 'soft_match_questions', 'exact_matched_answers', 'soft_match_answers'])

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
