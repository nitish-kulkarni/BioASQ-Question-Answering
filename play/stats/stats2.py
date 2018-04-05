import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import json
import cPickle as pickle

def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

stop_words = set(stopwords.words('english'))


def check_stop_noun(w, pos):

	return ((not w in stop_words) and (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'))

def get_nouns(query):

	sentences = query['sentences']
	candidates = []
	nouns_per_sent = []

	for sentence in sentences:
		words_pos = nltk.pos_tag(word_tokenize(sentence))
		nouns = [w.lower() for w,pos in words_pos if check_stop_noun(w,pos)]
		nouns_per_sent.append(nouns)
		candidates += nouns

	return candidates, nouns_per_sent

def get_answer_nouns(query):
	answers = query['answer'][0]
	if type(answers) != type(list()):
		answers = [answers]

	set_nouns = set()
	for answer in answers:
		words_pos = nltk.pos_tag(word_tokenize(answer))
		nouns = [w.lower() for w,pos in words_pos if check_stop_noun(w,pos)]

		for noun in nouns:
			set_nouns.add(noun)

	return list(set_nouns)


def main():
	queries = load_object('pkl/space.pkl')
	count = 0

	_objs = []
	for query in queries:
		candidates, nouns_per_sent = get_nouns(query)
		answer_nouns = get_answer_nouns(query)

		obj = {
		'question': query['query'],
		'sentences': nouns_per_sent,
		'candidates': candidates,
		'answers': answer_nouns
		}

		_objs.append(obj)

	save_object('pkl/prep.pkl', _objs)


if __name__ == '__main__':
	main()





