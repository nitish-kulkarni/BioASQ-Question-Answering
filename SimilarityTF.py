from SimilarityMeasure import *

class SimilarityTfidf(SimilarityMeasure):
	def calculateSimilarity(self):
		set1 = set([i.lower() for i in word_tokenize(self.sentence1) if i.lower() not in self.stopWords])
		set2 = set([i.lower() for i in word_tokenize(self.sentence2) if i.lower() not in self.stopWords])
		return float(len(set1.intersection(set2)))/len(set1.union(set2))
