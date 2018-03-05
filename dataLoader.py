"""Load, process and enrich the input data
"""

import json
import numpy as np
import pandas as pd

import ner.pubtator as pubtator
import re

class Question():

    def __init__(self, qid, q_type, question, documents, snippets, ideal_answer_ref=None, exact_answer_ref=None):

        self.qid = qid
        self.type = q_type
        self.question = question
        self.documents = documents
        self.snippets = snippets

        self.ideal_answer = None
        self.exact_answer = None
        self.ideal_answer_ref = ideal_answer_ref
        self.exact_answer_ref = exact_answer_ref

        self.V_snippets = _vocab_size([i.text for i in snippets])

class Snippet():

    def __init__(self, text, document_uri):

        self.text = text
        self.doc_id = _id_from_pubmed_uri(document_uri)

class Document():

    def __init__(self, document_uri):

        self.doc_id = _id_from_pubmed_uri(document_uri)
        self.text = ''
        # self.text = pubtator.get_doctext_from_docid(self.doc_id)

class DataLoader():

    def __init__(self, input_path):

        with open(input_path, 'r') as fp:
            self.data = json.load(fp)['questions']
            self.questions = self._get_questions()

    def get_DF(self):
        return pd.DataFrame(self.data)

    def get_questions_of_type(self, qtype):
        return [question for question in self.questions if question.type == qtype]

    def _get_questions(self):
        questions = []
        for qid, question in enumerate(self.data):
            snippets = [Snippet(s['text'], s['document']) for s in question.get('snippets', [])]
            documents = [Document(doc_uri) for doc_uri in question.get('documents', [])]

            questions.append(
                Question(
                    qid,
                    question['type'],
                    question['body'],
                    documents,
                    snippets,
                    ideal_answer_ref=question.get('ideal_answer', None),
                    exact_answer_ref=question.get('exact_answer', None)
                )
            )

        return questions

def _id_from_pubmed_uri(uri):
    return int(uri.split('/')[-1])

def _tokenize(text):
    return re.findall('\w+', text)

def _vocab_size(strings):
    if len(strings) == 0:
        return 0
    return np.unique(np.concatenate([_tokenize(string) for string in strings])).size
