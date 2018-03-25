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


def preprocess_data(data):

    processed_data = []
    for (i, question) in enumerate(data['questions']):
        processed_question = LeToR(question)
        processed_data.append(processed_question)

    return processed_data


def get_golden_ranking(question):
    ideal_answer = question['ideal_answer']
    sentences = RM.get_sentences(question['snippets'])
    sentences = RM.preprocess_sentences(sentences)
    ranked_sentences = RM.get_ranked_sentences(question_text= ideal_answer, sentences=sentences, retrieval_algo='BM25')
    return ranked_sentences


def filter_summary_type_questions(data):
    fp = open('train_5b_summary.json', 'wb')
    summary_questions = list(filter(lambda x: x['type'] == 'summary', data['questions']))
    json.dump(summary_questions, fp)
    fp.close()


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

    print ('Done Preprocessing !!')

if __name__ == '__main__':
    main()