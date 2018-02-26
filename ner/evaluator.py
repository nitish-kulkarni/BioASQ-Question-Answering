"""Evaluate a NER/bioconcept tagger
"""

import re
import numpy as np
import pandas as pd

def _entities_from_tagger(tagger, doc_id, doc_text):
    return [e['entity'].lower() for e in tagger(doc_id, doc_text)]

def evaluate(tagger, data):
    list_type = data.get_questions_of_type('list')
    factoid_type = data.get_questions_of_type('factoid')

    results = []
    for question in list_type + factoid_type:
        answers = [answer.lower() for answer in _flatten(question.exact_answer_ref)]
        entities = []
        for document in question.documents:
            entities += _entities_from_tagger(tagger, document.doc_id, document.text)
        entities = list(np.unique(entities))
        exact_matches = sum([answer in entities for answer in answers])
        soft_matches = _soft_matches(answers, entities)
        results.append({
            'answers': answers,
            'entities': entities,
            'exact_matches': exact_matches,
            'total_answers': len(answers),
            'total_entities': len(entities),
            'soft_matches': soft_matches,
            'type': question.type
        })

    return pd.DataFrame(results)

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

