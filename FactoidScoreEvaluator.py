
from nltk.tokenize import sent_tokenize
import retrieval_model as RM
import json
import time


def create_index(sentences):
    term_dict = dict()
    for i, sentence in enumerate(sentences):
        tokens = RM.get_tokens(sentence)
        term_dict = RM.update_dictionary(term_dict, tokens, i)

    return term_dict


def get_sentences(snippets):
    sentences = []
    snippetsText = []
    for snippet in snippets:
        text = unicode(snippet.text).encode("ascii", "ignore")
        snippetsText.append(text)
        if text == "":
            continue
        try:
            sentences += sent_tokenize(text)
        except:
            sentences += text.split(". ")  # Notice the space after the dot
    return sentences


def getScoreList(question_text, snippets):
    sentences = get_sentences(snippets)
    sentences = set(sentences)
    sentences = RM.preprocess_sentences(sentences)

    scorelist = []
    inverted_index = create_index(sentences)
    question_tokens = RM.get_tokens(question_text)

    N = len(sentences)
    avg_length = RM.get_average_sentence_length(inverted_index, N)

    for i, sentence in enumerate(sentences):
        sentence_score = dict()
        sentence_tokens = RM.get_tokens(sentence)
        doc_length = len(sentence_tokens)
        indri_score = RM.get_Indri_Score(inverted_index=inverted_index, question_tokens=question_tokens,
                                       docId=i, doc_length=doc_length, N=N)

        bm25_score = RM.get_BM25_score(term_dict=inverted_index, question_tokens=question_tokens,
                                      docId=i, tokens=sentence_tokens, N=N, avg_doc_length=avg_length)

        sentence_score['sentence'] = sentence
        sentence_score['indri'] = indri_score
        sentence_score['bm25'] = bm25_score

        scorelist.append(sentence_score)

    return scorelist


def main():

    # Dummy code for test

    filepath = './input/toydata.json'

    fp = open(filepath)
    data = json.load(fp)

    for (i, question) in enumerate(data['questions']):

        score = getScoreList(question['body'], question['snippets'])


# if __name__ == '__main__':
#     main()