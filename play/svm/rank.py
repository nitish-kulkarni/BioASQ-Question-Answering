from data import train_rank, test_rank
import svmlight
import operator
import cPickle as pickle

def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

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

class SVMRank():

	def __init__(self, weights=None):
		self.weights = weights
		self.feed_X = []
		self.feed_y = []
		self.feed_ids = []

		if not self.weights:
			self.model = svmlight

	def feed(self, X, y, ids):
		self.feed_X += X
		self.feed_y += y
		self.feed_ids += ids

	def train_from_feed(self):
		formatted_X = []
		X, y, ids = self.feed_X, self.feed_y, self.feed_ids

		for _X, _y, _id in zip(X, y, ids):
			light_format = SVMLightFormat(_y, _id, _X).get()
			formatted_X.append(light_format)

		self._train(formatted_X)

	def _train(self, formatted_X):
		self.current = self.model.learn(formatted_X, type='ranking', verbosity=0)

	def train(self, X, y, ids):

		formatted_X = []
		for _X, _y, _id in zip(X, y, ids):
			light_format = SVMLightFormat(_y, _id, _X).get()
			formatted_X.append(light_format)

		self._train(formatted_X)

	def evaluate(self, X, y, ids):

		formatted_X = []
		for _X, _y, _id in zip(X, y, ids):
			light_format = SVMLightFormat(_y, _id, _X).get()
			formatted_X.append(light_format)

		return self._evaluate(formatted_X)

	def _evaluate(self, formatted_X):

		predictions = self.model.classify(self.current, formatted_X)
		ranked_arr = []

		for entry, score in zip(formatted_X, predictions):
			_info = (entry[0], score, entry[2])
			ranked_arr.append(_info)

		ranked_arr.sort(key=operator.itemgetter(1))
		ranked_result = []
		for i in range(len(ranked_arr)):
			rank_obj = {
			'ground_truth': ranked_arr[i][0],
			'ranking': i+1,
			'id': ranked_arr[i][2]
			}
			ranked_result.append(rank_obj)

		return ranked_result

	def save(self, name):
		self.model.write_model(self.current, name)

def main():

	train_X = load_object('pkl/train_X')
	train_y = load_object('pkl/train_y')
	train_ids = load_object('pkl/train_ids')

	test_X = load_object('pkl/test_X')
	test_y = load_object('pkl/test_y')
	test_ids = load_object('pkl/test_ids')


	ranker = SVMRank()
	ranker.train(train_X, train_y, train_ids)
	results = ranker.evaluate(test_X, test_y, test_ids)
	print results

if __name__ == "__main__":
	main()



"""
train_X = [
    (3, [(1, 1.0), (2, 1.0), (3, 0.0), (4, 0.2), (5, 0.0)], 1), 
    (2, [(1, 0.0), (2, 0.0), (3, 1.0), (4, 0.1), (5, 1.0)], 1), 
    (1, [(1, 0.0), (2, 1.0), (3, 0.0), (4, 0.4), (5, 0.0)], 1), 
    (1, [(1, 0.0), (2, 0.0), (3, 1.0), (4, 0.3), (5, 0.0)], 1), 
    (1, [(1, 0.0), (2, 0.0), (3, 1.0), (4, 0.2), (5, 0.0)], 2), 
    (2, [(1, 1.0), (2, 0.0), (3, 1.0), (4, 0.4), (5, 0.0)], 2), 
    (1, [(1, 0.0), (2, 0.0), (3, 1.0), (4, 0.1), (5, 0.0)], 2), 
    (1, [(1, 0.0), (2, 0.0), (3, 1.0), (4, 0.2), (5, 0.0)], 2), 
    (2, [(1, 0.0), (2, 0.0), (3, 1.0), (4, 0.1), (5, 1.0)], 3), 
    (3, [(1, 1.0), (2, 1.0), (3, 0.0), (4, 0.3), (5, 0.0)], 3), 
    (4, [(1, 1.0), (2, 0.0), (3, 0.0), (4, 0.4), (5, 1.0)], 3), 
    (1, [(1, 0.0), (2, 1.0), (3, 1.0), (4, 0.5), (5, 0.0)], 3) ]


test_X = [
    (4, [(1, 1.0), (2, 0.0), (3, 0.0), (4, 0.2), (5, 1.0)], 4), 
    (3, [(1, 1.0), (2, 1.0), (3, 0.0), (4, 0.3), (5, 0.0)], 4), 
    (2, [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.2), (5, 1.0)], 4), 
    (1, [(1, 0.0), (2, 0.0), (3, 1.0), (4, 0.2), (5, 0.0)], 4) ]

X, y, _ids = [], [], []
for el in train_X:
	_y, _x, _id = el

	x_arr = []
	for _ in _x:
		x_arr.append(_[1])

	X.append(x_arr)
	y.append(_y)
	_ids.append(_id)

save_object('pkl/train_X', X)
save_object('pkl/train_y', y)
save_object('pkl/train_ids', _ids)


X, y, _ids = [], [], []
for el in test_X:
	_y, _x, _id = el

	x_arr = []
	for _ in _x:
		x_arr.append(_[1])

	X.append(x_arr)
	y.append(_y)
	_ids.append(_id)

save_object('pkl/test_X', X)
save_object('pkl/test_y', y)
save_object('pkl/test_ids', _ids)
"""

"""
#ranker = SVMRank()
#ranker._train(train_rank)
#ranker._evaluate(test_rank)

 
#model = svmlight.learn(train_rank, type='ranking', verbosity=0)
#svmlight.write_model(model, 'test.dat')

#evaluate(test_rank)
"""

