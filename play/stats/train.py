import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import json
import cPickle as pickle
from collections import defaultdict as dd
import operator

def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def check_stop_noun(w, pos):
	stop_words = set(stopwords.words('english'))
	return ((not w in stop_words) and ((pos == 'NN') or (pos == 'NNP') or (pos == 'NNS') or (pos == 'NNPS') or (pos == 'JJ')))

def get_nouns_from_sentence(sentence):

	words_pos = nltk.pos_tag(word_tokenize(sentence))
	nouns = [w.lower() for w,pos in words_pos if check_stop_noun(w,pos)]

	return nouns

def main():
	
	queries = load_object('pkl/factoid-6b.pkl')
	#query = queries[0]
	#feature_vec = vectorize_query(query)

	average = []
	pos = dd(int)

	for query in queries:
		answer = query['answer'][0]
		answer_words = word_tokenize(answer)
		nouns = get_nouns_from_sentence(answer)

		div = len(nouns)/float(len(answer_words))
		average.append(div)

		if div<1:
			answer_words = [w.lower() for w in answer_words]
			word_pos = nltk.pos_tag(answer_words)
			print nltk.pos_tag(answer_words)
			print nltk.pos_tag(nouns)
			words_eval = [w for w in answer_words if w not in nouns]
			print words_eval
			print '---\n\n'

			for el in nltk.pos_tag(words_eval):
				p = el[1]
				pos[p] += 1


	pos_sorted = sorted(pos.items(), key=operator.itemgetter(1), reverse=True)
	print pos_sorted



	print np.array(average).mean()

if __name__ == '__main__':
	main()