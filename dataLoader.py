"""Load, process and enrich the input data
"""

import re
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import ner.pubtator as pubtator
from ner.lingpipe import NER_tagger_multiple
from nltk.tokenize import sent_tokenize

from constants import *

class Question():

    def __init__(self,
        qid, q_type, question,
        documents, snippets,
        ideal_answer_ref=None,
        exact_answer_ref=None
    ):

        self.qid = qid
        self.type = q_type
        self.question = question
        self.documents = documents
        self.snippets = snippets
        self.snippet_sentences = _get_sentences([i.text for i in self.snippets])

        self.ideal_answer = None
        self.exact_answer = None
        self.ideal_answer_ref = ideal_answer_ref
        self.exact_answer_ref = exact_answer_ref

        self.ner_entities = []
        self.V_snippets = _vocab_size([i.text for i in snippets])

        # positive assertion for the question
        self.assertion_pos = None
        self.assertion_neg = None

    def get_ner_entity_list(self):
        return [e['entity'] for e in self.ner_entities]

    def ranked_sentences():


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
            self.data = json.load(fp)[QUESTIONS]
        self.questions = self._get_questions()
        self.name = input_path.split('/')[-1].strip('.json')

        _ensure_path('nerCache')
        self.ner_cache_filename = 'nerCache/%s.json' % self.name

    def __getitem__(self, idx):
        return self.questions[idx]

    # should be the same as __getitem__
    def question_from_qid(self, qid):
        return [q for q in self.questions if q.qid == qid][0]

    def get_DF(self):
        return pd.DataFrame(self.data)

    def get_questions_of_type(self, qtype):
        return [question for question in self.questions if question.type == qtype]

    def load_ner_entities(self):

        list_type = self.get_questions_of_type(LIST_TYPE)
        factoid_type = self.get_questions_of_type(FACTOID_TYPE)
        questions = list_type + factoid_type
        # questions = questions[:10]

        if os.path.exists(self.ner_cache_filename):
            print('Loading ner entities from file: %s' % self.ner_cache_filename)
            with open(self.ner_cache_filename, 'r') as fp:
                ner_dict = json.load(fp)
            for q in questions:
                key = str(q.qid)
                if key in ner_dict:
                    q.ner_entities = ner_dict[key]
        else:
            _load_ner(questions, pubtator.get_bio_concepts_multiple, PUBTATOR, multiple=True)
            _load_ner(questions, NER_tagger_multiple, LINGPIPE, multiple=True, snippets=True)
            for q in questions:
                q.ner_entities = _unique(q.ner_entities)

            print('Saving ner entities to file: %s' % self.ner_cache_filename)
            with open(self.ner_cache_filename, 'w') as fp:
                json.dump(_ner_dict(questions), fp)

    def save_ners(self, output_file):
        data = {}
        questions = []
        self.load_ner_entities()
        for q in self.questions:
            elem = {}
            elem[NER_ENTITIES] = q.ner_entities
            elem[BODY] = q.question
            elem[TYPE] = q.type
            elem[EXACT_ANSWER] = q.exact_answer_ref
            elem[IDEAL_ANSWER] = q.ideal_answer_ref
            questions.append(elem)
        data[QUESTIONS] = questions
        with open(output_file, 'w') as fp:
            json.dump(data, fp, indent=4, sort_keys=True)

    def _get_questions(self):
        questions = []
        for qid, question in enumerate(self.data):
            snippets = [Snippet(s['text'], s['document']) for s in question.get('snippets', [])]
            documents = [Document(doc_uri) for doc_uri in question.get('documents', [])]

            questions.append(
                Question(
                    qid,
                    question[TYPE],
                    question[BODY],
                    documents,
                    snippets,
                    ideal_answer_ref=question.get(IDEAL_ANSWER, None),
                    exact_answer_ref=question.get(EXACT_ANSWER, None)
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

def _load_ner(questions, tagger, tagger_name, multiple=False, snippets=False):
    missed = 0
    print('Loading {0} entiies..'.format(tagger_name))
    for question in tqdm(questions):
        docs = question.snippets if snippets else question.documents
        try:
            if multiple:
                doc_ids = [document.doc_id for document in docs]
                doc_texts = [document.text for document in docs]
                entities = _entities_from_tagger(tagger, doc_ids, doc_texts)
            else:
                entities = []
                for document in docs:
                    entities += _entities_from_tagger(tagger, document.doc_id, document.text)
        except:
            missed += 1
            continue
        new_entities = []
        for entity in entities:
            entity[SOURCE] = tagger_name
            new_entities.append(entity)
        question.ner_entities += new_entities
    print('Failed to load {0} entities for {1} questions'.format(tagger_name, missed))
    print('Finished loading {0} entiies..'.format(tagger_name))

def _entities_from_tagger(tagger, doc_id, doc_text):
    return [e for e in tagger(doc_id, doc_text)]

def _unique(entities):
    types = {}
    for e in entities:
        if ENTITY not in e or TYPE not in e:
            continue
        types[e[ENTITY].lower()] = (e[TYPE], e[SOURCE])
    return [{ENTITY: k, TYPE: v, SOURCE: s} for k, (v, s) in types.items()]

def _flatten(l):
    if not isinstance(l, list):
        return [l]
    flattened = []
    for item in l:
        flattened += _flatten(item)
    return flattened

def _ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _ner_dict(questions):
    d = {}
    for q in questions:
        d[q.qid] = q.ner_entities
    return d

def _get_sentences(snippets):
    sentences = []
    for snippet in snippets:
        text = unicode(snippet).encode("ascii", "ignore")
        if text == '':
            continue
        try:
            sentences += sent_tokenize(text)
        except:
            sentences += text.split(". ")  # Notice the space after the dot
    return sentences
