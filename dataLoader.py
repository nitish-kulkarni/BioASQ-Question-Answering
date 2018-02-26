"""Load, process and enrich the input data
"""

import json
import ner.pubtator as pubtator

class Question():

    def __init__(self, q_type, question, documents, snippets):

        self.type = q_type
        self.question = question
        self.documents = documents
        self.snippets = snippets

class Snippet():

    def __init__(self, text, document_uri):

        self.text = text
        self.doc_id = _id_from_pubmed_uri(document_uri)

class Document():

    def __init__(self, document_uri):

        self.doc_id = _id_from_pubmed_uri(document_uri)
        # self.text = pubtator.get_doctext_from_docid(self.doc_id)

class DataLoader():

    def __init__(self, input_path):

        with open(input_path, 'r') as fp:
            self.data = json.load(fp)['questions']
            self.questions = self._get_queries()

    def _get_queries(self):
        questions = []
        for question in self.data:
            snippets = [Snippet(s['text'], s['document']) for s in question.get('snippets', [])]
            documents = [Document(doc_uri) for doc_uri in question.get('documents', [])]

            questions.append(
                Question(
                    question['type'],
                    question['body'],
                    documents,
                    snippets,
                )
            )

        return questions

def _id_from_pubmed_uri(uri):
    return int(uri.split('/')[-1])