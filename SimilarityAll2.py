import re, math
from collections import Counter

from SimilarityMeasure import *
from SimilarityCosine import *
from SimilarityDice import *
from SimilarityJaccard import *
from Root import *


class SimilarityAll2(SimilarityMeasure):

    def calculateSimilarity(self):
        jac = SimilarityJaccard
        cos = SimilarityCosine
        dice = SimilarityDice
        root = Root

        rootres = root(self.sentence1, self.sentence2).calculateSimilarity()
        result =  jac(self.sentence1,self.sentence2).calculateSimilarity()+\
        cos(self.sentence1,self.sentence2).calculateSimilarity()+\
        dice(self.sentence1,self.sentence2).calculateSimilarity()+\
        rootres
        return result

