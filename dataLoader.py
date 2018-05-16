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
import retrieval_model
from constants import *

class Question():

    def __init__(self,
        qid, q_type, question,
        documents, snippets, bioasq_id,
        ideal_answer_ref=None,
        exact_answer_ref=None
    ):

        self.qid = qid
        self.bioasq_id = bioasq_id
        self.type = q_type
        self.question = question
        self.documents = documents
        self.snippets = snippets
        self.snippet_sentences = _get_sentences([i.text for i in self.snippets])
        self.snippet_blob = '. '.join(self.snippet_sentences)

        self.ideal_answer = None
        self.exact_answer = None
        self.ideal_answer_ref = ideal_answer_ref
        self.exact_answer_ref = exact_answer_ref

        self.ner_entities = []
        self.snippet_ner_entities = []
        self.V_snippets = _vocab_size([i.text for i in snippets])

        # positive assertion for the question
        self.assertion_pos = None
        self.assertion_neg = None

    def get_ner_entity_list(self):
        return [e['entity'] for e in self.ner_entities]

    def _snippet_ner_entities(self):
        return [e for e in self.ner_entities if e[ENTITY] in self.snippet_blob]

    def ranked_sentences(self):
        scores = {}
        for sentence, score in retrieval_model.get_ranked_sentences(self.question, self.snippet_sentences, BM25):
            scores[sentence] = [score]
        for sentence, score in retrieval_model.get_ranked_sentences(self.question, self.snippet_sentences, INDRI):
            scores[sentence].append(score)

        all_sentences = [(bm25_score, indri_score, s) for s, [bm25_score, indri_score] in scores.items()]
        all_sentences = sorted(all_sentences, reverse=True)
        sentences = [{
            TEXT: s,
            BM25: bm25_score,
            INDRI: indri_score
        } for (bm25_score, indri_score, s) in all_sentences]
        return sentences

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

        for q in questions:
            q.snippet_ner_entities = q._snippet_ner_entities()

    def eval_yesno(self):
        yesno = self.get_questions_of_type(YESNO_TYPE)
        yes_correct = 0.0
        no_correct = 0.0
        total_yes = 0.0
        total_no = 0.0

        for q in yesno:
            if not q.exact_answer_ref or not q.exact_answer:
                continue
            if q.exact_answer_ref == 'yes':
                total_yes += 1
                yes_correct += (q.exact_answer_ref == q.exact_answer)
            if q.exact_answer_ref == 'no':
                total_no += 1
                no_correct += (q.exact_answer_ref == q.exact_answer)

        total_correct = yes_correct + no_correct
        total_q = total_yes + total_no
        total_acc = total_correct / total_q
        yes_acc = yes_correct / total_yes
        no_acc = no_correct / total_no
        print('Total Accuracy = %2f (%d / %d)' % (total_acc, total_correct, total_q))
        print('YES Accuracy = %2f (%d / %d)' % (yes_acc, yes_correct, total_yes))
        print('No Accuracy = %2f (%d / %d)' % (no_acc, no_correct, total_no))

    def eval_factoid(self):
        factoids = self.get_questions_of_type(FACTOID_TYPE)
        mrrs = []
        precisions = []
        soft_precisions = []
        missing = 0
        for q in factoids:
            rank = 1
            mrr = 0
            precision = 0
            soft_precision = 0
            if not q.exact_answer:
                missing += 1
                continue
            exact_answer_ref = [i for i in np.array(q.exact_answer_ref).flatten()]
            # exact_answer_ref = [i.lower() for i in np.array(q.exact_answer_ref).flatten()]
            for answer in q.exact_answer:
                # if answer.lower() in exact_answer_ref:
                if answer in exact_answer_ref:
                    mrr = 1.0 / rank
                    if rank == 1:
                        precision = 1.0
                    soft_precision = 1.0
                    break
                rank += 1
            mrrs.append(mrr)
            precisions.append(precision)
            soft_precisions.append(soft_precision)
        print('Missing answers for %d questions' % missing)
        print('Precision: %.2f ' % (np.mean(precisions) * 100) )
        print('Soft Precision: %.2f ' % (np.mean(soft_precisions) * 100) )
        print('MRR: %.2f ' % (np.mean(mrrs) * 100) )

    def load_answers_from_file(self, filename, exact):
        with open(filename, 'r') as fp:
            filedata = json.load(fp)
        
        for qid, value in filedata.items():
            if value is not None:
                q = self.questions[int(qid)]
                if exact:
                    q.exact_answer = value
                else:
                    q.ideal_answer = value

    def save_submission(self, input_file, output_file):
        with open(input_file, 'r') as fp:
            data = json.load(fp)
        for q in self.questions:
            assert q.type == data[QUESTIONS][q.qid][TYPE]
            assert q.bioasq_id == data[QUESTIONS][q.qid][ID]
            if q.exact_answer:
                data[QUESTIONS][q.qid][EXACT_ANSWER] = q.exact_answer
            else:
                if q.type == YESNO_TYPE:
                    data[QUESTIONS][q.qid][EXACT_ANSWER] = 'yes'
                if q.type in [FACTOID_TYPE, LIST_TYPE]:
                    data[QUESTIONS][q.qid][EXACT_ANSWER] = []
            data[QUESTIONS][q.qid][IDEAL_ANSWER] = q.ideal_answer if q.ideal_answer else 'yes'
        with open(output_file, 'w') as fp:
            json.dump(data, fp, indent=4, sort_keys=True)

    def save_yesno_answers(self, filename):
        answers = {}
        yesno = self.get_questions_of_type(YESNO_TYPE)
        for q in yesno:
            answers[q.qid] = q.exact_answer
        with open(filename, 'w') as fp:
            json.dump(answers, fp, indent=4, sort_keys=True)

    def save_factoid_list_answers(self, filename):
        answers = {}
        factoids = self.get_questions_of_type(FACTOID_TYPE)
        listtype = self.get_questions_of_type(LIST_TYPE)
        for q in factoids + listtype:
            answers[q.qid] = q.exact_answer
        with open(filename, 'w') as fp:
            json.dump(answers, fp, indent=4, sort_keys=True)

    def save_ideal_answers(self, filename):
        answers = {}
        for q in self.questions:
            answers[q.qid] = q.ideal_answer
        with open(filename, 'w') as fp:
            json.dump(answers, fp, indent=4, sort_keys=True)

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
                    question[ID],
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
    sentences = np.unique(sentences).tolist()
    sentences = [i.strip().strip('.') for i in sentences]
    return sentences
