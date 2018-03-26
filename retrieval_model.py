import json
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import operator

stop_words = set(stopwords.words('english'))


class BM25(object):
    def __init__(self, k1, k3, b):
        self.k1 = k1
        self.k3 = k3
        self.b = b

    def get_score(self, term_dict, common_tokens, N, docId, average_doc_length):
        score = 0
        doc_length = get_doc_length(term_dict, docId)
        for token in common_tokens:
            score += self.get_individual_term_score(term_dict, token, N, doc_length, average_doc_length)
        return score

    def get_individual_term_score(self, term_dict, token, N, doc_length, average_doc_length):
        df_term = len(term_dict[token])
        df_term = (N - df_term + 0.5) / (df_term + 0.5)
        idf = np.log(1 + df_term)

        tf = sum([term_dict[token][x] for x in term_dict[token].keys()])

        tf /= tf + self.k1 * ((1 - self.b) + self.b * (doc_length / average_doc_length))

        return tf * idf


class Indri(object):

    def __init__(self, lambda_, mu):
        self.lambda_ = lambda_
        self.mu = mu

    def get_score(self, inverted_index, question_tokens, doc_length, docId, N):
        score = 1.0
        for token in question_tokens:
            temp = self.get_individual_term_score(inverted_index, token, doc_length, docId, N)
            score *= temp

        score = np.power(score, float(1)/len(question_tokens))
        return score

    def get_individual_term_score(self, inverted_index, token, doc_length, docId, N):
        p_mle = 1.0 * max(np.sum(inverted_index.get(token, {}).values()), 1) / N
        score = self.lambda_ * p_mle
        score += (1 - self.lambda_) * (inverted_index.get(token, {}).get(docId, 0) + self.mu * p_mle)/ (doc_length + self.mu)
        return score


def get_sentences(snippets):
    sentences = []
    snippetsText = []
    for snippet in snippets:
        text = unicode(snippet["text"]).encode("ascii", "ignore")
        snippetsText.append(text)
        if text == "":
            continue
        try:
            sentences += sent_tokenize(text)
        except:
            sentences += text.split(". ")  # Notice the space after the dot
    return sentences


def preprocess_sentences(sentences):
    cleaned_sentences = set()
    for sentence in sentences:
        s = sentence.rstrip().lstrip()
        s = s.replace('.', '')
        cleaned_sentences.add(s)
    return list(cleaned_sentences)


def get_average_sentence_length(index, N):
    total_terms = 0
    for i, term in enumerate(index):
        for j, doc in enumerate(index[term]):
            total_terms += index[term][doc]

    return float(total_terms)/N

def get_docID(snippet):
    doc_string = snippet['document']
    doc_id = int(doc_string.rstrip().replace('\'', '').split('/')[4])
    return doc_id


def update_dictionary(term_dict, tokens, docId):
    # updates the term dictionary for tokens of docId
    for token in tokens:
        if term_dict.get(token, False):
            if term_dict[token].get(docId, False):
                term_dict[token][docId] += 1
            else:
                term_dict[token][docId] = 1
        else:
            term_dict[token] = {docId: 1}

    return term_dict


def get_doc_length(term_dict, docId):
    count = 0
    for key in term_dict.keys():
        if docId in term_dict[key].keys():
            count += term_dict[key][docId]

    return count


def get_BM25_score(term_dict, question_tokens, docId, tokens, N, avg_doc_length):
    common_tokens = set(question_tokens).intersection(set(tokens))
    bm25_model = BM25(k1=1.2, k3=0, b=0.75)
    score = bm25_model.get_score(term_dict, common_tokens, N, docId, avg_doc_length)
    return score


def get_Indri_Score(inverted_index, question_tokens, docId, doc_length, N):
    indri_model = Indri(lambda_=0.75, mu=5000)
    score = indri_model.get_score(inverted_index, question_tokens, doc_length, docId, N)
    return score


def get_ranked_snippets(snippets, question_text, algo='BM25'):
    question_tokens = get_tokens(question_text)
    term_dict = dict()

    ranked_snippets = []
    snippet_text_dict = dict()
    total_token_count = 0
    for snippet in snippets:

        tokens = get_tokens(snippet['text'])
        total_token_count += len(tokens)
        term_dict = update_dictionary(term_dict, tokens, snippet['text'].encode('ascii', "ignore"))
        snippet_text_dict[snippet['text'].encode('ascii', "ignore")] = snippet

    score_list = dict()
    N = len(snippets)
    avg_doc_length = float(total_token_count) / N

    for snippet in snippets:
        tokens = get_tokens(snippet['text'])
        # score = get_BM25_score(term_dict, question_tokens, snippet['text'].encode('ascii', "ignore"), tokens, N, avg_doc_length)
        score = get_Indri_Score(term_dict, question_tokens, snippet['text'].encode('ascii', "ignore"), len(tokens))
        score_list[snippet['text'].encode('ascii', "ignore")] = score

    sorted_d = sorted(score_list.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(len(sorted_d)):
        ranked_snippets.append(snippet_text_dict[sorted_d[i][0]])

    return ranked_snippets


def get_score_list(filepath, algo='BM25'):
    fp = open(filepath)
    data = json.load(fp)

    question_score_list = dict()

    for (i, question) in enumerate(data['questions']):
        snippets = question['snippets']
        question_text = question['body']
        question_tokens = get_tokens(question_text)
        term_dict = dict()
        snippet_token_list = []
        total_token_count = 0
        for snippet in snippets:
            docId = get_docID(snippet)
            tokens = get_tokens(snippet['text'])
            total_token_count += len(tokens)
            term_dict = update_dictionary(term_dict, tokens, docId)
            snippet_token_list.append((docId, tokens))

        score_list = dict()
        N = len(snippets)
        avg_doc_length = float(total_token_count) / N
        for item in snippet_token_list:
            docId = item[0]
            tokens = item[1]
            score = get_BM25_score(term_dict, question_tokens, docId, tokens, N, avg_doc_length)
            score_list[docId] = score

        sorted(score_list.iteritems(), key=lambda (k, v): (v, k), reverse=True)
        question_score_list[i] = score_list
        break
    return question_score_list


def get_tokens(text):
    text = text.lower()
    text = text.encode('ascii', "ignore")
    sentences = sent_tokenize(text)
    tokens = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [w for w in words if not w in stop_words]
        words = [re.sub(r'\W+', '', w) for w in filtered_words]
        words = [w for w in words if w != '']
        tokens.extend(words)
    return tokens


def create_index(sentences):
    term_dict = dict()

    for i, sentence in enumerate(sentences):
        tokens = get_tokens(sentence)
        term_dict = update_dictionary(term_dict, tokens, i)

    return term_dict


# current

def get_ranked_sentences(question_text, sentences, retrieval_algo):

    # sentences = preprocess_sentences(sentences)
    N = len(sentences)
    inverted_index = create_index(sentences)
    avg_length = get_average_sentence_length(inverted_index, N)
    question_tokens = get_tokens(question_text)

    scorelist = dict()
    bm25_model = BM25(k1=1.2, k3=0, b=0.75)

    for i, sentence in enumerate(sentences):
        sentence_tokens = get_tokens(sentence)
        score = 0.0
        if retrieval_algo == 'BM25':
            common_tokens = set(question_tokens).intersection(set(sentence_tokens))
            score = bm25_model.get_score(term_dict=inverted_index, common_tokens=common_tokens, N=N,
                                         average_doc_length=avg_length, docId=i)

        if retrieval_algo == 'Indri':
            pass

        scorelist[sentence] = score

    sorted_sentences = sorted(scorelist.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_sentences


def main():
    filepath = './input/toydata.json'
    # score_list = get_score_list(filepath, 'BM25')
    print('retrieved results')


# if __name__ == '__main__':
#     main()