import logging
from logging import config
from BiRanker import BiRanker
import retrieval_model as RM
import operator
from SimilarityJaccard import *
from SimilarityDice import *
from nltk.tokenize import sent_tokenize

logging.config.fileConfig('logging.ini')
logger = logging.getLogger('bioAsqLogger')


class CustomSimilarMMR(BiRanker):

    def compute_Positions(self, snippets, question):

        pos_dict = {}
        max_rank = len(snippets)
        rank = 0

        snippets = RM.get_ranked_snippets(snippets, question_text=question['body'], algo='BM25')

        for snippet in snippets:
            snippet = unicode(snippet["text"]).encode("ascii", "ignore")
            more_sentences = [i.lstrip().rstrip() for i in sent_tokenize(snippet)]

            for sentence in more_sentences:
                if sentence not in pos_dict:
                    pos_dict[sentence] = 1 - (float(rank) / max_rank)
            rank += 1
        logger.info('Computed position dictionary for Bi Ranking')
        return pos_dict

    def getRankedList(self, question):

        selectedSentences = []
        snippets = question['snippets']

        # This is the class method from the BiRanker that is used to compute
        # the positional scores of the sentences in the snippets.

        pos_dict = self.compute_Positions(snippets, question)
        self.beta = 0.5
        best = []
        current_best = None

        # class method from abstract class that tokenizes all the snippets to sentences.

        sentences = self.getSentences(question)
        for i in range(self.numSelectedSentences):
            best_sim = -99999999
            for sentence in sentences:

                # similarityJaccard is an extension of Similarity Measure
                # that takes 2 sentences and returns the float (similarity)

                similarityInstance = SimilarityDice(sentence, question['body'])
                ques_sim = similarityInstance.calculateSimilarity()

                max_sent_sim = -99999999
                for other in best:
                    similarityInstance = SimilarityDice(sentence, other)
                    if self.beta != 0:
                        try:
                            current_sent_sim = (self.beta * similarityInstance.calculateSimilarity()) + (
                            (1 - self.beta) * pos_dict[sentence])
                        except:
                            logger.info(
                                'Looking for Sentence: ' + str(sentence.lstrip().rstrip()) + 'in positional dictionary')
                            current_sent_sim = (self.beta * similarityInstance.calculateSimilarity()) + (
                            (1 - self.beta) * pos_dict[sentence.lstrip().rstrip()])
                    else:  # since the value of beta is set to 0
                        current_sent_sim = similarityInstance.calculateSimilarity()
                    if current_sent_sim > max_sent_sim:
                        max_sent_sim = current_sent_sim

                # equation for mmr to balance between similarity with
                #  already selected sentences and similarity with question
                final_sim = ((1 - self.alpha) * ques_sim) - (self.alpha * max_sent_sim)
                if final_sim > best_sim:
                    best_sim = final_sim
                    current_best = sentence
            best.append(current_best)

            # maintaining a list of sentences that are not already
            # selected so they can be used for selection for next iteration
            sentences = set(sentences).difference(set(best))
            if current_best != None:
                selectedSentences.append(current_best)
            else:
                break
        logger.info('Performed Core MMR')
        return selectedSentences

