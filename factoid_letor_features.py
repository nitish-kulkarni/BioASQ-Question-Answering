"""Contains feature functions for LETOR
"""
import numpy as np


def _get_sentences_with_entity(sentences, entity):
    filtered_sentences = list(filter(lambda x: entity in x['sentence_body'], sentences))
    return filtered_sentences


def _BM25_score(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    score = reduce(lambda x, y: x['bm25_score']+y['bm25_score'], filtered_sentences)
    return score


def _indri_score(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    score = reduce(lambda x, y: x['indri_score']+y['indri_score'], filtered_sentences)
    return score


def _num_sentences(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    return len(filtered_sentences)


def _is_pubtator_type(question, entity_type):
    return question['type'] == entity_type


def _is_lingpipe_type(question, entity_type):
    return question['type'] == entity_type


def _tf_idf(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    df = len(filtered_sentences)
    N = len(sentences)
    idf = np.log(float(N + 1)/(df + 0.5))
    ctf = reduce(lambda x, y: x['sentence_body'].count(entity) + y['sentence_body'].count(entity), filtered_sentences)

    return ctf * idf


def _is_pubtator_entity(entity):
    pass


def _is_lingpipe_entity(entity):
    pass


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
    entity = 'Homo erectus'
    """
    entity_text = entity['entity']
    entity_type = entity['type']
    raw_sentences = [i['sentence_body'] for i in sentences]
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