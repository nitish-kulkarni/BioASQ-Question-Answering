# from flask import Flask, request, abort, render_template
# from flask import jsonify, render_template

import os
java_path = "C:/../../jdk1.8.0_101/bin/java.exe"
os.environ['JAVAHOME'] = java_path

import sys
import json
import copy
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

from Expander import Expander
from NoExpander import NoExpander
#from SnomedctExpander import SnomedctExpander
#from UMLSExpander import UMLSExpander

from BiRanker import BiRanker
from GreedyRanker import GreedyRanker
from coreMMR import CoreMMR
from SoftMMR import SoftMMR
from HardMMR import HardMMR
from BM25Ranker import BM25Ranker
from IndriRanker import IndriRanker

from Tiler import Tiler
from Concatenation import Concatenation

from Fusion import Fusion
import EvaluatePrecision

from KMeansSimilarityOrderer import KMeansSimilarityOrderer
from MajorityOrder import MajorityOrder
from MajorityCluster import MajorityCluster

from Evaluator import Evaluator
import pyrouge

import logging
from logging import config

#from pymetamap import MetaMap
# from singletonConceptId import *

import question_classifier

import retrieval_model as rm
import sys

from PyRouge import pyrouge as Pyrouge

'''
@Author: Khyathi Raghavi Chandu
@Date: October 17 2017

This code has the entire pipeline built from the classes to execute bioasq ideal answer generation.
Running this code results in a json file that can be directly uploaded on the oracle to get the official ROUGE scores.
Running the code:
$> python pipeline.py ./input/phaseB_4b_04.json > submission.json
'''

#logging.config.fileConfig('logging.ini')

'''
# create logger with 'spam_application'
logger = logging.getLogger('bioasq')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('bioAsq.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename = 'bioAsq.log',
    filemode = 'w'
)
'''

#logging.config.fileConfig('logging.ini')
logging.config.fileConfig('logging.ini')
logger = logging.getLogger('bioAsqLogger')


class Pipeline(object):
    def __init__(self, filePath, expanderInstance, biRankerInstance, orderInstance, fusionInstance, tilerInstance):
        self.filePath = filePath
        self.expanderInstance = expanderInstance
        self.biRankerInstance = biRankerInstance
        self.orderInstance = orderInstance
        self.fusionInstance = fusionInstance
        self.tilerInstance = tilerInstance

    def getSummaries(self):

        # metamapInstance = SingletonMetaMap.Instance()
        # metamapInstance.startMetaMap()
        # raw_input()

        rouge = Pyrouge.Rouge()

        allAnswerQuestion = []
        infile = open(self.filePath, 'r')
        data = json.load(infile)
        logger.info('Loaded training data')
        # qc = question_classifier.classifier()

        count = 0
        avg_prec = 0
        avg_recall = 0
        avg_f = 0

        ## split train and test data
        total_num_data = len(data['questions'])
        training_num_data = int(total_num_data*0.7)
        val_num_data = int(total_num_data*0.2)
        test_num_data = int(total_num_data*0.1)
        train_data = data['questions'][:training_num_data]
        dev_data = data['questions'][training_num_data:training_num_data+val_num_data]
        test_data = data['questions'][training_num_data+val_num_data:]
        print('# of training data:',len(train_data),', # of dev data:', len(dev_data),' # of test data:', len(test_data))
        
        test_id = 0
        for (i, question) in enumerate(test_data): # looping over all test questions
            logger.info('Started summarization pipeline for Question '+ str(i))

            ExpansiontoOriginal = {}
            SentencetoSnippet = {}

            #pred_cat = qc.classify(question['body'])
            logger.info('Generated question classification for the question')
            pred_length = 30
            pred_cat = question['type']
            if pred_cat=='summary':
                pred_length = 7
            elif pred_cat=='list':
                pred_length = 5
            elif pred_cat=='factoid':
                pred_length = 3
            elif pred_cat=='yesno':
                pred_length = 4
            else:
                pass

            #only consider summary question
            if pred_cat != 'summary':
                continue
            test_id += 1
            print('Working on test data num :',test_id)

            modifiedQuestion = copy.copy(question)

            logger.info('Performing expansions...')
            

            #EXECUTIONS OF EXPANSIONS
            #expansion on question body i.e, the text in the question
            expandedQuestion = self.expanderInstance.getExpansions(question['body'])

            #expansion on every sentence in each of the snippets
            expandedSnippets = []
            for snippet in question['snippets']:
                expandedSnippet = snippet
                expandedSentences = ""
                for sentence in sent_tokenize(snippet['text']):
                    expandedSentence = self.expanderInstance.getExpansions(sentence)
                    expandedSentences += expandedSentence + " "
                    ExpansiontoOriginal[expandedSentence.strip()] = sentence.strip()
                    SentencetoSnippet[sentence.strip()] = snippet
                expandedSnippet['text'] = expandedSentences
                expandedSnippets.append(expandedSnippet)

            modifiedQuestion['snippets'] = expandedSnippets
            modifiedQuestion['body'] = expandedQuestion
            logger.info('Updated the question with expander output...')

            custom_ranked_snippets = rm.get_ranked_snippets(question['snippets'], question['body'])
            question['snippets'] = custom_ranked_snippets

            #EXECUTION OF ONE OF BIRANKERS
            #rankedSentencesList = self.biRankerInstance.getRankedList(modifiedQuestion)
            rankedSentencesList = self.biRankerInstance.getRankedList(question)
            logger.info('Retrieved ranked list of sentences...')


            #ExpansiontoOriginal = {value: key for key, value in OriginaltoExpansion.iteritems()}
            rankedSentencesListOriginal = []
            rankedSnippets = []
            for sentence in rankedSentencesList:
                try:
                    rankedSentencesListOriginal.append(ExpansiontoOriginal[sentence.strip()])
                    rankedSnippets.append(SentencetoSnippet[sentence.strip()])
                except:
                    pass

            #EXECUTION OF TILING
            tiler_info = {'max_length': 200, 'max_tokens': 200, 'k': 2, 'max_iter': 20}
            try:
                orderedList = self.orderInstance.orderSentences(rankedSentencesListOriginal, rankedSnippets, tiler_info)
            except:
                print question['body']
            fusedList = self.fusionInstance.tileSentences(orderedList, 200)
            logger.info('Tiling sentences to get alternative summary...')
            
            #EXECUTION OF EVAULATION (To be done)
            #evaluatorInstance = Evaluator()
            #goldIdealAnswer, r2, rsu = evaluatorInstance.calculateRouge(question['body'], finalSummary)

            #uncomment the following 3 lines for fusion
            concat_inst = Concatenation()
            #finalSummary = concat_inst.tileSentences(rankedSentencesList, pred_length) #pred_length*5
            finalSummary = concat_inst.tileSentences(fusedList, 200) #pred_length*5
            #baseline_summary = concat_inst.tileSentences(rankedSentencesListOriginal, pred_length)
            #finalSummary = EvaluatePrecision.betterAnswer(baseline_summary, fused_Summary, question['body'])

            #logger.info('Choosing better summary ...')
            if pred_cat == 'summary':
                try:
                    score = rouge.rouge_l([finalSummary.encode('ascii',errors='ignore')], [str(question['ideal_answer']).encode('ascii',errors='ignore')])
                except:
                    print('shit')
                    continue
                #score = rouge.rouge_l([unicode(finalSummary,errors='ignore')], [unicode(str(question['ideal_answer']),errors='ignore')])
                count += 1
                avg_prec += score[0]
                avg_recall += score[1]
                avg_f += score[2]
            question['ideal_answer'] = finalSummary

            AnswerQuestion = question
            allAnswerQuestion.append(AnswerQuestion)
            logger.info('Inserted ideal answer into the json dictionary')
        # metamapInstance.stopMetaMap()
        avg_prec /= count
        avg_recall /= count
        avg_f /= count
        print 'My results '
        print avg_prec, avg_recall, avg_f
        print('finish testing...')
        quit()
        return allAnswerQuestion

if __name__ == '__main__':

    # filePath = sys.argv[1]
    filePath = "./input/BioASQ-trainingDataset5b.json"
    expanderInstance = NoExpander()
    biRankerInstance = CoreMMR()
    #biRankerInstance = GreedyRanker()
    # biRankerInstance = BM25Ranker()
    # biRankerInstance = IndriRanker()
    orderInstance = MajorityCluster()
    fusionInstance = Fusion()
    tilerInstance = Concatenation()
    #tilerInstance = MajorityOrder()
    #tilerInstance = KMeansSimilarityOrderer()
    pipelineInstance = Pipeline(filePath, expanderInstance, biRankerInstance, orderInstance, fusionInstance ,tilerInstance)
    #pipelineInstance = Pipeline(filePath)
    idealAnswerJson = {}
    idealAnswerJson['questions'] = pipelineInstance.getSummaries()
    with open('ordered_fusion_indri_latest_5b_test.json', 'w') as outfile:
        json.dump(idealAnswerJson, outfile)
    print json.dumps(idealAnswerJson)
