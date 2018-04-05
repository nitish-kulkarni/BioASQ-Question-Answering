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
	factoid_queries = load_object('pkl/factoid_queries.pkl')
	print len(factoid_queries)
	_factoid_queries = [q for q in factoid_queries if len(q['sentences'])>0]
	print len(factoid_queries)
	factoid_queries = [q for q in _factoid_queries if len(q['answer'])<2]
	print len(factoid_queries)

	occ = []
	no_exact = []
	for query in _factoid_queries:
		answers = query['answer'][0]
		sentences = query['sentences']

		if type(answers) != type(list()):
			answers = [answers]

		#answer = answers[0]
		status = 0
		occ.append(status)

		for answer in answers:
			#print answer.lower()
			for sentence in sentences:
				if answer.lower() in sentence.lower():
					occ.pop()
					status = 1
					occ.append(status)
					break

		if status == 0:
			no_exact.append(query)

	print np.array(occ).mean(), 'aqui'

	#print no_exact[-3]
	occ = []
	for query in no_exact:
		answers = query['answer'][0]
		sentences = query['sentences']

		if type(answers) != type(list()):
			answers = [answers]

		answer = answers[0]
		tokens = [w for w,pos in nltk.pos_tag(word_tokenize(answer)) if ((not w in stop_words) and (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'))]

		status = 0
		occ.append(status)
		for sentence in sentences:
			for answer in tokens:
				if answer.lower() in sentence.lower():
					occ.pop()
					status = 1
					occ.append(status)
					break

	
	print np.array(occ).mean(), 'aca'

	occ = []
	no_exact = []
	all_nouns = []
	for query in _factoid_queries:
		answers = query['answer'][0]
		sentences = query['sentences']

		if type(answers) != type(list()):
			answers = [answers]

		for answer in answers:
			nouns = [w for w,pos in nltk.pos_tag(word_tokenize(answer)) if ((not w in stop_words) and (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'))]

			all_nouns.append(len(nouns))

			status = 0
			occ.append(status)
			for sentence in sentences:
				for answer in nouns:
					if answer.lower() in sentence.lower():
						occ.pop()
						status = 1
						occ.append(status)
						break

			if status == 0:
				no_exact.append(query)

	print np.array(occ).mean(), 'pair'
	print np.array(all_nouns).mean(), 'pair'

	founds = []
	query_space = []
	for query in _factoid_queries:
		answers = query['answer'][0]
		sentences = query['sentences']

		if type(answers) != type(list()):
			answers = [answers]

		found_all = False
		for answer in answers:
			nouns = [w for w,pos in nltk.pos_tag(word_tokenize(answer)) if ((not w in stop_words) and (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'))]
			total_nouns = len(nouns)
			count_of_nouns = 0
			used_nouns = set()

			for noun in nouns:
				for sentence in sentences:
					if noun.lower() in sentence.lower():
						count_of_nouns+=1
						#print noun, nouns
						break

			if ((total_nouns == count_of_nouns) and (total_nouns>0)):
				found_all = True
			else:
				#print count_of_nouns/float(total_nouns)
				pass

		if found_all:
			if len(answers)<=1:
				founds.append(1)
				query_space.append(query)
		else:
			founds.append(0)

	print np.array(founds).mean(), 'all'
	save_object('pkl/space.pkl', query_space)
	#print np.array(occ).mean()
	#print np.array(all_nouns).mean()



if __name__ == '__main__':
	main()