"""Contains feature functions for LETOR
"""

def _BM25_score(sentences, entity):
    pass

def _indri_score(sentences, entity):
    pass

def _num_sentences(sentences, entity):
    pass

def _is_pubtator_type(question, entity_type):
    pass

def _is_lingpipe_type(question, entity_type):
    pass

def _tf_idf(sentences, entity):
    pass

def _is_pubtator_entity(entity):
    pass

def _is_lingpipe_entity(entity):
    pass

def all_features(question, sentences, entity):
    """
    question = 'What causes Hiershpriung disease?'
    sentences = [
        ('The gene abra ka dabra causes Hiershpriung disease', bm25_score, indri_score),
        ('I really don't know', bm25_score, indri_score),
    ]
    entity = 'Homo erectus'
    """
    entity_text = entity['entity']
    entity_type = entity['type']
    raw_sentences = [i[0] for i in sentences]
    features = [
        _BM25_score(sentences, entity_text),
        _indri_score(sentences, entity_text),
        _num_sentences(raw_sentences, entity_text),
        _is_pubtator_type(question, entity_type),
        _is_lingpipe_type(question, entity_type),
        _tf_idf(raw_sentences, entity_text),
        _is_pubtator_entity(entity),
        _is_lingpipe_entity(entity),
    ]
    return features
