from nltk.parse.stanford import StanfordDependencyParser
from SimilarityMeasure import *
import spacy
from nltk.stem.lancaster import LancasterStemmer

class Root(SimilarityMeasure):
	def calculateSimilarity(self):
		en_nlp = spacy.load('en')
		st = LancasterStemmer()
		question = self.sentence1.lower()
		question_root = st.stem(str([sent.root for sent in en_nlp(question.decode('utf8')).sents][0]))

		sentence = en_nlp(self.sentence2.lower().decode('utf8'))
		roots = [st.stem(chunk.root.head.text.lower()) for chunk in sentence.noun_chunks]
		if question_root in roots:
			return 0.1
		else:
			return 0.0
