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


def main():
	queries = load_object('pkl/prep.pkl')
	count_arr = []

	"""
	for query in queries:
		question = query['question']
		answers = query['answers']
		sentences = query['sentences']
		candidates = query['candidates']

		for answer in answers:
			count_appearance = 0

			for sentence in sentences:
				if answer in sentence:
					count_appearance+=1

			count_arr.append(count_appearance/float(len(sentences)))
	"""

	position_arr = []
	c_arr = []
	for query in queries:
		question = query['question']
		answers = query['answers']
		sentences = query['sentences']
		candidates = query['candidates']

		c_arr.append(len(candidates))

		freq_dict = {}
		for candidate in candidates:
			doc_freq = 0

			for sentence in sentences:
				if candidate in sentence:
					doc_freq += 1

			freq_dict[candidate] = doc_freq
			_freq = sorted([(k,v) for k,v in freq_dict.iteritems()], key=lambda x: x[1], reverse=True)
			_freq = [k for k,v in _freq]

		for answer in answers:
			if answer in _freq:
				position = _freq.index(answer) + 1
				position_arr.append(position/float(len(_freq)))

	print (np.array(position_arr) > 0.23).sum() / float(len(position_arr))

	print (np.array(c_arr).mean())



	


if __name__ == '__main__':
	main()