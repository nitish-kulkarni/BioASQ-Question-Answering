import numpy as np
from nltk.tokenize import sent_tokenize
import retrieval_model as RM
import json


class LeToR(object):
    def __init__(self, question):
        self.initialize_params(question)

    def initialize_params(self, question):
        # It is a list of answers (could range from 1 to n)
        self.ideal_answer = question['ideal_answer']
        sentences = RM.get_sentences(question['snippets'])
        self.sentences = RM.preprocess_sentences(sentences)
        self.question_type = question['type']
        self.question_text = unicode(question['body']).encode("ascii", "ignore")


def preprocess_data(data):

    processed_data = []
    for (i, question) in enumerate(data):
        processed_question = LeToR(question)
        processed_data.append(processed_question)

    return processed_data


def get_golden_ranking(question):
    ideal_answer = question['ideal_answer']
    sentences = RM.get_sentences(question['snippets'])
    sentences = RM.preprocess_sentences(sentences)
    ranked_sentences = RM.get_ranked_sentences(question_text=ideal_answer, sentences=sentences, retrieval_algo='BM25')
    return ranked_sentences


def filter_summary_type_questions(data):
    fp = open('./input/train_5b_summary.json', 'wb')
    summary_questions = list(filter(lambda x: x['type'] == 'summary', data['questions']))
    json.dump(summary_questions, fp)
    fp.close()


def create_feature_vectors(question):

    sentences = set(question.sentences)
    feature_vectors = []

    # ranked sentences also gives a score which can be used as feature

    ranked_sentences_bm25 = RM.get_ranked_sentences(question_text=question.question_text, sentences=sentences,
                                                    retrieval_algo='BM25')

    # ranked sentences also gives a score which can be used as feature

    ranked_sentences_Indri = RM.get_ranked_sentences(question_text=question.question_text, sentences=sentences,
                                                     retrieval_algo='Indri')

    # TO DO @Gabe, above results can be used for LeToR

    return feature_vectors


def main():

    filepath = './input/BioASQ-trainingDataset5b.json'
    fp = open(filepath)
    train_data = json.load(fp)
    fp.close()

    filter_summary_type_questions(train_data)

    fp = open('./input/train_5b_summary.json')
    data = json.load(fp)
    fp.close()

    processed_data = preprocess_data(data)
    for i, processed_question in enumerate(processed_data):
        if len(processed_question.sentences) == 0:
            continue
        vv = create_feature_vectors(processed_question)

    print ('Done Preprocessing !!')

if __name__ == '__main__':
    main()