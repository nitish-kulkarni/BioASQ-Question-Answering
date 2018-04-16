#local
from svm.rank import SVMRank
from modules.dataLoader import DataLoader
import modules.constants as C
import modules.factoid_letor_features as factoid_letor_features

#global
import numpy as np
import json
from sklearn.metrics import classification_report
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict as dd
from collections import OrderedDict as od
from nltk.corpus import stopwords
import cPickle as pickle
import wikiwords
stop_words = set(stopwords.words('english'))

def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def overlap_score(entity1, entity2):
    if entity1 == '':
        return 0.0

    words1 = entity1.split(' ')
    words2 = entity2.split(' ')
    size1 = float(len(set(words1)))
    size2 = float(len(set(words2)))

    if size1 * size2 == 0:
        return 0.0

    overlap = float(len(set(words1).intersection(set(words2))))
    return (overlap * overlap) / (size1 * size2)

def gold_candidate_rank(candidates, gold_answers):
    scores = []
    candidates = [candidate[C.ENTITY] for candidate in candidates]
    for candidate in candidates:
        score = np.array([overlap_score(candidate.lower().strip(), answer.lower().strip()) for answer in gold_answers]).max()
        scores.append((score, candidate))
    ranks = {}
    scores = sorted(scores, reverse=True)
    for rank, (score, candidate) in enumerate(scores):
        ranks[candidate] = rank
    return [ranks[candidate] for candidate in candidates]

def get_features_old(question, ranked_sentences, candidates, candidate_extra_features):
    # candidates = question.snippet_ner_entities
    candidates = [{
        C.ENTITY: candidate,
        C.TYPE: '',
        C.SOURCE: '',
    } for candidate in candidates]
    X = np.array([factoid_letor_features.all_features(question.question, ranked_sentences, candidate) for candidate in candidates])
    gold_answers = question.exact_answer_ref[0]
    if type(gold_answers) != type(list()):
        gold_answers = [gold_answers]

    y = gold_candidate_rank(candidates, gold_answers)
    return X.tolist(), y

def get_features(query, sentences, candidates, gold_answers):

    candidates_obj = []

    #prepare
    plain_sentences = [s['text'] for s in sentences]
    w_dict = dd(int)
    tokenizer = RegexpTokenizer(r'\w+')
    for sentence in plain_sentences:
        finder_n(tokenizer.tokenize(sentence.lower()), w_dict)

    total_count = 0
    for key in w_dict:
        total_count += w_dict[key]

    #iterate
    for candidate in candidates:
        candidate_obj = {}
        candidate_obj['candidate'] = candidate
        features = []
        #features += _BM25_score(sentences, candidate)
        #features += _indri_score(sentences, candidate)
        #features += _tf_idf(sentences, candidate)
        #features += _num_sentences(sentences, candidate)
        features += _basic_features(sentences, candidate, w_dict, total_count, candidates, query)
        candidate_obj['features'] = features
        candidate_obj['score'] = np.array([overlap_score(candidate.lower().strip(), answer.lower().strip()) for answer in gold_answers]).max()
        candidates_obj.append(candidate_obj)

    return candidates_obj

def finder_n(words, w_dict):
    words = [a for a in words if a not in stop_words]
    size = len(words)
    for k in range(1, 5):
        for i in range(size):
            w_dict[' '.join(words[i:i+k])] += 1
    return []

def get_candidates(sentences_obj):
    sentences = [s['text'] for s in sentences_obj]
    w_dict = dd(int)
    tokenizer = RegexpTokenizer(r'\w+')
    for sentence in sentences:
        finder_n(tokenizer.tokenize(sentence.lower()), w_dict)
        #finder_n(sentence.lower().split(), w_dict)

    n_relevant = [(k,v) for k,v in dict(w_dict).iteritems()]
    n_relevant.sort(key=lambda x: x[1], reverse=True)
    n_relevant = n_relevant[:100]

    candidates = [k for k,v in n_relevant]

    return candidates

def get_answer_arr(answers_exact):

    answers = answers_exact[0]

    if type(answers) != type(list()):
        answers = [answers]

    return [ans.lower() for ans in answers]

def _get_sentences_with_entity(sentences, entity):
    filtered_sentences = list(filter(lambda x: entity.lower() in x[C.TEXT].lower(), sentences))
    return filtered_sentences


def _BM25_score(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    score = np.array([i[C.BM25] for i in filtered_sentences]).sum()
    return [score]


def _indri_score(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    score = np.array([i[C.INDRI] for i in filtered_sentences]).sum()
    return [score]


def _num_sentences(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    return [float(len(filtered_sentences))]

def _tf_idf(sentences, entity):
    filtered_sentences = _get_sentences_with_entity(sentences, entity)
    df = len(filtered_sentences)
    N = len(sentences)
    idf = np.log(float(N + 1)/(df + 0.5))
    ctf = np.array([i[C.TEXT].count(entity) for i in filtered_sentences]).sum()

    return [ctf * idf]

def _basic_features(sentences, entity, w_dict, total_count, candidates, query):

    position = w_dict[entity]/float(total_count)
    ov = 0.0

    for other_candidate in candidates:
        if entity in other_candidate:
            ov += 1.0
    ov = ov/float(len(candidates))

    is_in_query = 0.0

    if entity in query:
        is_in_query = 1.0



    return [position, is_in_query]

def main():

    questions_obj = []
    file_name = 'input/BioASQ-trainingDataset5b.json'
    data = DataLoader(file_name)
    data.load_ner_entities()

    questions = data.get_questions_of_type(C.FACTOID_TYPE)
    for i, question in enumerate(tqdm(questions)):
        query = question.question.lower()
        answers_arr = get_answer_arr(question.exact_answer_ref)
        sentences_obj = question.ranked_sentences()
        candidates = get_candidates(sentences_obj)
        candidates_obj = get_features(query, sentences_obj, candidates, answers_arr)

        question_obj = {
        'sentences': sentences_obj,
        'candidates': candidates_obj,
        'query': query,
        'answers': answers_arr
        }
        questions_obj.append(question_obj)

    save_object('pkl/questions.pkl', questions_obj)


if __name__ == '__main__':
    main()