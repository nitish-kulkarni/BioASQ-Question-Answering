import pickle
import question_category as qc
from nltk import word_tokenize
import numpy as np

class classifier(object):
	"""docstring for ClassName"""
	def __init__(self):
		#Load vectorizers and models
		self.ngram_vec = pickle.load(open("model/ngram_vec.pickle",'rb'))
		self.hw_vec = pickle.load(open("model/head_word_vec.pickle",'rb'))
		self.clf = pickle.load(open("model/question_classifier.pickle",'rb'))
	def gen_feature(self, new_q):
		ngram = (self.ngram_vec.transform(qc.first_n_gram([new_q],4)).toarray())
		head_word = (self.hw_vec.transform(qc.extract_head_word([new_q])).toarray())
		
		ques_token = word_tokenize(new_q)
		wh = qc.extract_ques_word(ques_token) #Presence of wh words etc.
		pattern = qc.extract_pattern(new_q) #Question pattern
		
		feature_space = []
	
		other_feaeture = []
		other_feaeture.extend(wh)
		other_feaeture.extend(pattern)
		other_feaeture = np.array(other_feaeture).reshape(1,len(other_feaeture))
		feature_space = np.c_[ngram,np.array(head_word), other_feaeture]
		return feature_space
	def classify(self,new_q):
		X = self.gen_feature(new_q)
		pred = self.clf.predict(X)
		return pred[0]

		
