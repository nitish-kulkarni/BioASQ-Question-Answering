import re, math
from collections import Counter

from SimilarityMeasure import *
from SimilarityCosine import *
from SimilarityDice import *
from SimilarityJaccard import *


class SimilarityAll(SimilarityMeasure):


    def calculateSimilarity(self):
        jac = SimilarityJaccard
        cos = SimilarityCosine
        dice = SimilarityDice
        return jac(self.sentence1,self.sentence2).calculateSimilarity()+\
        cos(self.sentence1,self.sentence2).calculateSimilarity()+dice(self.sentence1,self.sentence2).calculateSimilarity()