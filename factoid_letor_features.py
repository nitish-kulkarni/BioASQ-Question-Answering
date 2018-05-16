"""Contains feature functions for LETOR
"""
import numpy as np
import constants as C

def _get_sentences_with_entity(sentences, entity):
    filtered_sentences = list(filter(lambda x: entity.lower() in x[C.TEXT].lower(), sentences))
    return filtered_sentences


def _BM25_score(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    score = np.array([i[C.BM25] for i in filtered_sentences]).sum()
    return score


def _indri_score(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    score = np.array([i[C.INDRI] for i in filtered_sentences]).sum()
    return score


def _num_sentences(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    return float(len(filtered_sentences))


def _is_pubtator_type(question, entity_type):
    return float(entity_type.lower() in question.lower())


def _is_lingpipe_type(question, entity_type):
    return float(entity_type.lower() in question.lower())


def _tf_idf(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    df = len(filtered_sentences)
    N = len(sentences)
    idf = np.log(float(N + 1)/(df + 0.5))
    ctf = np.array([i[C.TEXT].count(entity) for i in filtered_sentences]).sum()

    return ctf * idf


def _is_pubtator_entity(entity):
    return float(entity[C.SOURCE] == C.PUBTATOR)


def _is_lingpipe_entity(entity):
    return float(entity[C.SOURCE] == C.LINGPIPE)


def all_features(question, sentences, entity):
    """
    question = 'What causes Hiershpriung disease?'
    sentences = [
        {
        sentence_body:'The gene abra ka dabra causes Hiershpriung disease',
        bm25_score: bm25_val,
         indri_score: indri_val
        },
        {
        sentence_body:'I really don't know',
        bm25_score: bm25_val,
         indri_score: indri_val
        }
    ]
    entity = {
        'entity': 'Homo erectus',
        'type': 'Species',
        'source': 'PubTator'
    }
    """
    entity_text = entity[C.ENTITY]
    entity_type = entity[C.TYPE]
    raw_sentences = [i[C.TEXT] for i in sentences]
    features = [
        _BM25_score(sentences, entity_text),
        _indri_score(sentences, entity_text),
        _num_sentences(sentences, entity_text),
        _is_pubtator_type(question, entity_type),
        _is_lingpipe_type(question, entity_type),
        _tf_idf(sentences, entity_text),
        _is_pubtator_entity(entity),
        # _is_lingpipe_entity(entity),
    ]
    return features


def main():
    entity = {}
    entity['text'] = 'Homo erectus'
    entity['type'] = 'gene'
    question = {}
    question['text'] = 'What causes Hiershpriung disease?'
    question['type'] = 'gene'
    sentences = [
        {
            'sentence_body': 'The Homo erectus abra ka dabra causes Hiershpriung disease',
            'bm25_score': 3.2,
            'indri_score': 0.4
        },

        {
            'sentence_body': 'I really gene dont know about gene',
            'bm25_score': 2.5,
            'indri_score': 0.3
        },

        {
            'sentence_body': 'I dont know this stupid Homo erectus enzyme',
            'bm25_score': 4.5,
            'indri_score': 0.6
        }

    ]

    print  _tf_idf(sentences, entity['text'])


if __name__ == '__main__':
    main()