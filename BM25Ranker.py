import logging
from logging import config
from BiRanker import BiRanker
import retrieval_model as RM
import operator

logging.config.fileConfig('logging.ini')
logger = logging.getLogger('bioAsqLogger')


def create_index(sentences):
    term_dict = dict()
    for i, sentence in enumerate(sentences):
        tokens = RM.get_tokens(sentence)
        term_dict = RM.update_dictionary(term_dict, tokens, i)

    return term_dict


def get_average_sentence_length(index, N):
    total_terms = 0
    for i, term in enumerate(index):
        for j, doc in enumerate(index[term]):
            total_terms += index[term][doc]

    return float(total_terms)/N


def preprocess_sentences(sentences):
    cleaned_sentences = set()
    for sentence in sentences:
        s = sentence.rstrip().lstrip()
        s = s.replace('.', '')
        cleaned_sentences.add(s)

    return list(cleaned_sentences)


class BM25Ranker(BiRanker):

    def getRankedList(self, question):

        selectedSentences = []
        sentences = self.getSentences(question)
        sentences = set(sentences)

        sentences = preprocess_sentences(sentences)

        N = len(sentences)

        scorelist = dict()

        inverted_index = create_index(sentences)
        avg_length = get_average_sentence_length(inverted_index, N)
        question_tokens = RM.get_tokens(question['body'])

        for i, sentence in enumerate(sentences):
            sentence_tokens = RM.get_tokens(sentence)
            score = RM.get_BM25_score(term_dict=inverted_index, question_tokens=question_tokens,
                                      docId=i, tokens=sentence_tokens, N=N, avg_doc_length=avg_length)

            scorelist[sentence] = score

        sorted_d = sorted(scorelist.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(self.numSelectedSentences):
            selectedSentences.append(sorted_d[i][0])
        return selectedSentences
