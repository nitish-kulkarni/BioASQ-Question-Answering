from svm.rank import SVMRank
from dataLoader import DataLoader
import constants as C
import factoid_letor_features
import numpy as np
import json
from sklearn.metrics import classification_report

from tqdm import tqdm

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

def get_scores(res):

    prediction = []
    ground_truth = []
    for r in res:
        pred, ground = r
        prediction.append(pred)
        ground_truth.append(ground)

    return classification_report(prediction, prediction)


def gold_candidate_rank(candidates, gold_answers):
    scores = []
    candidates = [candidate[C.ENTITY] for candidate in candidates]
    for candidate in candidates:
        score = np.array([overlap_score(candidate, answer) for answer in gold_answers]).max()
        scores.append((score, candidate))
    ranks = {}
    scores = sorted(scores, reverse=True)
    for rank, (score, candidate) in enumerate(scores):
        ranks[candidate] = rank
    return [ranks[candidate] for candidate in candidates]

def get_features(question, ranked_sentences):
    candidates = question.snippet_ner_entities
    X = np.array([factoid_letor_features.all_features(question.question, ranked_sentences, candidate) for candidate in candidates])
    y = gold_candidate_rank(candidates, question.exact_answer_ref)
    return X.tolist(), candidates

def get_only_features(question, ranked_sentences):
    candidates = question.snippet_ner_entities
    X = np.array([factoid_letor_features.all_features(question.question, ranked_sentences, candidate) for candidate in candidates])
    candidates = [candidate[C.ENTITY] for candidate in candidates]
    return X.tolist(), candidates

def main():
    file_name = 'input/BioASQ-task6bPhaseB-testset3.json'
    file_name = 'input/BioASQ-trainingDataset6b.json'
    save_model_file_name = 'weights_2'
    ranker = SVMRank(save_model_file_name)
    data = DataLoader(file_name)
    data.load_ner_entities()
    ans_file = 'output/factoid_list_%s.json' % data.name

    questions = data.get_questions_of_type(C.FACTOID_TYPE)
    for i, question in enumerate(tqdm(questions)):
        ranked_sentences = question.ranked_sentences()
        X, candidates = get_only_features(question, ranked_sentences)
        top_answers = ranker.classify_from_feed(X, candidates, i)
        # question.exact_answer = [[answer] for answer in top_answers[:5]]
        question.exact_answer = [answer for answer in top_answers[:5]]
        # print question.exact_answer_ref
        # print '\n'
        # print top5
        # print '\n'
        # print '\n\n\n'
    # questions = data.get_questions_of_type(C.LIST_TYPE)
    # for i, question in enumerate(tqdm(questions)):
    #     ranked_sentences = question.ranked_sentences()
    #     X, candidates = get_only_features(question, ranked_sentences)
    #     top_answers = ranker.classify_from_feed(X, candidates, i)
    #     question.exact_answer = [[answer] for answer in top_answers[:10]]

    # data.save_factoid_list_answers(ans_file)
    data.eval_factoid()        

if __name__ == '__main__':
    main()