import svmlight
import cPickle as pickle
import letor as letor
import numpy as np
from nltk.tokenize import sent_tokenize
import retrieval_model as RM
import json
from collections import defaultdict as dd

class SVMLightFormat():

	def __init__(self, golden_rank, unique_id, feature_vec):
		self.golden_rank = int(golden_rank)
		self.unique_id = int(unique_id)
		self.feature_vec = feature_vec

		self.rank_vector = self.get()

	def get(self):
		_pairs = [(int(k[0]+1), float(k[1])) for k in enumerate(self.feature_vec)]
		_label = self.golden_rank
		_qid = self.unique_id
		_vector = (_label, _pairs, _qid)

		return _vector

def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def filter_data(filepath):

	fp = open(filepath)
	data = json.load(fp)
	fp.close()
	letor.filter_summary_type_questions(data)

	return './input/train_5b_summary.json'

def format_data(filepath):

	fp = open(filepath)
	data = json.load(fp)
	fp.close()

	_formatted = []
	processed_data = letor.preprocess_data(data)
	for i, processed_question in enumerate(processed_data):
		if len(processed_question.sentences) == 0:
			continue
		_f = letor.create_feature_vectors(processed_question)
		_r = letor.get_golden_ranking(processed_question)
		_formatted.append({'features': _f, 'ranking': _r})

	save_object('pkl/formatted_data', _formatted)
	return _formatted

def light_format(data):

	light_data = []
	unique_id = 1
	for point in data:
		ordered_sentences = point['ranking']
		features_group = [dict(f) for f in point['features']]


		rank = 1
		for sentence, score in ordered_sentences:
			features = [f[sentence] for f in features_group]
			light_tup = SVMLightFormat(rank, unique_id, features)
			light_data.append(light_tup.rank_vector)
			rank += 1

		unique_id+=1

	return light_data

def evaluation(data):

	_dict = dd(int)

	for el in data:
		_dict[el[2]].append(1)


def main():

	train_data = load_object('pkl/formatted_data')
	train_queries = light_format(train_data)

	#print train_queries[0]

	#evaluation(train_queries[:1])

	"""
	#training data
	try:
		train_data = load_object('pkl/formatted_data')
	except:
		#filtering
		file_train_path = './input/BioASQ-trainingDataset5b.json'
		filtered_train_path = filter_data(file_train_path)
		train_data = format_data(filtered_train_path)

	train_queries = light_format(train_data)
	model = svmlight.learn(train_queries, type='ranking', verbosity=0)
	svmlight.write_model(model, 'my_model.dat')

	#training data
	try:
		test_data = load_object('pkl/formatted_data')
	except:
		#filtering
		file_test_path = './input/BioASQ-trainingDataset5b.json'
		filtered_test_path = filter_data(file_test_path)
		test_data = format_data(filtered_test_path)


	test_queries = light_format(test_data)
	predictions = svmlight.classify(model, test_queries)
	for p in predictions:
		print '%.8f' % p
	"""

if __name__ == "__main__":
	main()




