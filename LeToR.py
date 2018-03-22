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


def main():

    # training data is available at the same path BioASQ-trainingDataset5b.json

    filepath = './input/toydata.json'
    fp = open(filepath)
    train_data = json.load(fp)

    processed_data = preprocess_data(train_data)

    print ('Done Preprocessing !!')

if __name__ == '__main__':
    main()