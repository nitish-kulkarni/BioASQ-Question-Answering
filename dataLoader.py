"""Load, process and enrich the input data
"""

import json

class Question():

    def __init__(self, q_type, question, documents, snippets):

        self.type = q_type
        self.question = question
        self.documents = documents
        self.snippets = snippets

class DataLoader():

    def __init__(self, input_path):

        with open(input_path, 'r') as fp:
            self.data = json.load(input_path)

    def get_questions(self):
        questions = []
        for question in self.data:
            questions.append(
                Question(
                    question['type'],
                    question['body'],
                    question['documents'],
                    question['snippets'],
                )
            )

        return questions
