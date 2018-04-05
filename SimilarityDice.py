from SimilarityMeasure import *


class SimilarityDice(SimilarityMeasure):
    def calculateSimilarity(self):
        a = self.sentence1
        b = self.sentence2
        if not len(a) or not len(b): return 0.0
        if len(a) == 1:
            a = a + u'.'
        if len(b) == 1:
            b = b + u'.'

        a_bigram_list = []
        for i in range(len(a) - 1):
            a_bigram_list.append(a[i:i + 2])
        b_bigram_list = []
        for i in range(len(b) - 1):
            b_bigram_list.append(b[i:i + 2])

        a_bigrams = set(a_bigram_list)
        b_bigrams = set(b_bigram_list)
        overlap = len(a_bigrams & b_bigrams)
        dice_coeff = overlap * 2.0 / (len(a_bigrams) + len(b_bigrams))
        return dice_coeff


"""
instance = SimilarityJaccard("apple banana cat dog", "apple elephant cat dog")
print instance.calculateSimilarity()
"""
