import modules.constants as C
import modules.factoid_letor_features
import numpy as np
import json
import nltk
from nltk import word_tokenize, RegexpTokenizer
from collections import defaultdict as dd
from collections import OrderedDict as od
import cPickle as pickle
from nltk.corpus import stopwords
import itertools as it
import modules.retrieval_model as RM
from get_features import _basic_features
from modules.dataLoader import DataLoader

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
        score = np.array([overlap_score(candidate.lower(), answer.lower()) for answer in gold_answers]).max()
        scores.append((score, candidate))
    ranks = {}
    scores = sorted(scores, reverse=True)
    for rank, (score, candidate) in enumerate(scores):
        ranks[candidate] = rank + 1
    return [ranks[candidate] for candidate in candidates]

# def get_features(question, ranked_sentences):
#     sentences = [a['text'].lower() for a in ranked_sentences]
#     candidates = get_top_entities(sentences)
#     X = np.array([factoid_letor_features.all_features(question.question, ranked_sentences, candidate) for candidate in candidates])
#     y = gold_candidate_rank(candidates, question.exact_answer_ref)
#     return X.tolist(), y

def finder_n(words, w_dict):

    words = [a for a in words if a not in stop_words]
    size = len(words)

    for k in range(1, 5):
        for i in range(size):
            w_dict[' '.join(words[i:i+k])] += 1
    return []

def get_top_entities(sentences):

    w_dict = dd(int)
    tokenizer = RegexpTokenizer(r'\w+')

    for sentence in sentences:
        finder_n(tokenizer.tokenize(sentence.lower()), w_dict)

    n_relevant = [(k,v) for k,v in dict(w_dict).iteritems()]
    n_relevant.sort(key=lambda x: x[1], reverse=True)

    return n_relevant

def get_features(query, sentences, candidates, gold_answers, ner_entities):

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
        features += _basic_features(sentences, candidate, w_dict, total_count, candidates, query, ner_entities)
        candidate_obj['features'] = features
        candidate_obj['score'] = np.array([overlap_score(candidate.lower().strip(), answer.lower().strip()) for answer in gold_answers]).max()
        candidates_obj.append(candidate_obj)

    return candidates_obj

def get_plain_features(query, sentences, candidates, ner_entities):

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
        features += _basic_features(sentences, candidate, w_dict, total_count, candidates, query, ner_entities)
        candidate_obj['features'] = features
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

# def _basic_features(sentences, entity, w_dict, total_count, candidates, query):
#     return 
#     position = w_dict[entity]/float(total_count)
#     ov = 0.0

#     for other_candidate in candidates:
#         if entity in other_candidate:
#             ov += 1.0
#     ov = ov/float(len(candidates))

#     is_in_query = 0.0

#     if entity in query:
#         is_in_query = 1.0



#     return [position, is_in_query]


def main():

    #file_name = 'input/BioASQ-trainingDataset5b.json'
    file_name = 'input/phaseB_5b_05.json'
    file_name = 'input/BioASQ-task6bPhaseB-testset5.json'
    questions = json.load(open(file_name))['questions']
    get_perc = []

    data = DataLoader(file_name)
    # data.load_ner_entities()
    qs = data.questions

    for i, question in enumerate(questions):

        if question['type'] == 'summary':
            questions[i]['ideal_answer'] = 'bla'

        if question['type'] == 'yesno':
            questions[i]['ideal_answer'] = 'bla'
            questions[i]['exact_answer'] = 'yes'
            

        if question['type'] == 'factoid':
            query = question['body'].lower() 
            sentences = RM.get_sentences(question['snippets'])
            sentences = RM.preprocess_sentences(sentences)
            sentences_obj = [{'text': s} for s in sentences]
            candidates = get_candidates(sentences_obj)
            candidates_obj = get_plain_features(query, sentences_obj, candidates, qs[i].snippet_ner_entities)

            clf = load_object('pkl/new_clf')
            candidates_arr = []
            for candidate in candidates_obj:
                features = np.array([np.array(candidate['features'])])
                score = clf.predict(features)[0]
                candidates_arr.append((score, candidate['candidate'], features))

            candidates_arr.sort(key=lambda x: x[0], reverse=True)
            candidates_arr = candidates_arr[:5]


            questions[i]['exact_answer'] = [[c] for s,c,f in candidates_arr]
            print questions[i]['exact_answer']
            questions[i]['ideal_answer'] = 'bla'

        if question['type'] == 'list':

            query = question['body'].lower() 
            sentences = RM.get_sentences(question['snippets'])
            sentences = RM.preprocess_sentences(sentences)
            sentences_obj = [{'text': s} for s in sentences]
            candidates = get_candidates(sentences_obj)
            candidates_obj = get_plain_features(query, sentences_obj, candidates, qs[i].snippet_ner_entities)

            clf = load_object('pkl/new_clf')
            candidates_arr = []
            for candidate in candidates_obj:
                features = np.array([np.array(candidate['features'])])
                score = clf.predict(features)[0]
                candidates_arr.append((score, candidate['candidate'], features))

            candidates_arr.sort(key=lambda x: x[0], reverse=True)
            candidates_arr = candidates_arr[:10]


            questions[i]['exact_answer'] = [[c] for s,c,f in candidates_arr]
            print questions[i]['exact_answer']
            questions[i]['ideal_answer'] = 'bla'



    #print np.array(get_perc).mean()
    with open('ans6b-full.json', 'w') as outfile:
        json.dump({'questions': questions}, outfile, indent=4)






    """
    ranker = SVMRank()
    file_name = 'input/BioASQ-trainingDataset6b.json'
    data = DataLoader(file_name)
    data.load_ner_entities()
    questions = data.get_questions_of_type(C.FACTOID_TYPE)[:419]

    for i, question in enumerate(questions):
        ranked_sentences = question.ranked_sentences()
        X, y = get_features(question, ranked_sentences)
        ranker.feed(X, y, i)

    ranker.train_from_feed()
    ranker.save('weights_new')
    """

    """
    file_name = 'input/BioASQ-trainingDataset6b.json'
    data = DataLoader(file_name)
    data.load_ner_entities()
    questions = data.get_questions_of_type(C.FACTOID_TYPE)
    get_perc = []
    count = 0


    for i, question in enumerate(questions):
        query = question.question.lower()
        sentences = [a['text'].lower() for a in question.ranked_sentences()]
        n_relevant = get_top_entities(sentences)

        answers = [a.lower() for a in question.exact_answer_ref]
        is_inside = False
        for el in n_relevant:
            for ans in answers:
                if el[0].lower() == ans.lower():
                    is_inside = True

        #print '\n\n'
        #print is_inside

        if is_inside:
            get_perc.append(1)
        else:
            get_perc.append(0)

        #print '\n\n'

    print np.array(get_perc).mean()
    """




if __name__ == "__main__":
    main()