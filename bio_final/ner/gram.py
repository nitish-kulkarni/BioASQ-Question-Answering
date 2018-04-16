#!/usr/bin/env python

import os
import numpy as np
import re
import optparse
import itertools
from collections import OrderedDict
import nltk
from gensim.models import word2vec

from cnngram.src import loader
from cnngram.src.utils2 import create_input
from cnngram.src.utils2 import models_path, evaluate, eval_script, eval_temp, reload_mappings, create_result, create_JNLPBA_result
from cnngram.src.loader import word_mapping, char_mapping, tag_mapping, pt_mapping
from cnngram.src.loader import update_tag_scheme, prepare_dataset
from cnngram.src.loader import augment_with_pretrained
from cnngram.src.GRAMCNN import GRAMCNN

import tensorflow as tf
import cPickle as pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_PATH = './ner'

def evaluate_sentence(doc_id, sentence, concept='all'):

    tokens = nltk.word_tokenize(sentence)
    test_sentences = [[[unicode(w), unicode('O')] for w in tokens]]

    test_data, m3 = prepare_dataset(
            test_sentences, word_to_id, char_to_id, tag_to_id, pt_to_id,lower
    )

    arr_results = evaluate(parameters, gramcnn, test_sentences,
                                      test_data, id_to_tag, remove = False, max_seq_len = max_seq_len, padding = parameters['padding'], use_pts = parameters['pts'])

    return process_results(arr_results, sentence)

def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def load_gramcnn():
    #load parameters
    #print '------params----'
    opts, parameters, model_name = load_object('%s/cnngram/src/main_params.pkl' % BASE_PATH)


    #prep for gram-cnn
    #print '------gram-cnn params----'
    lower = parameters['lower']
    zeros = parameters['zeros']
    tag_scheme = parameters['tag_scheme']
    word_to_id, char_to_id, tag_to_id, pt_to_id, dico_words, id_to_tag = reload_mappings('%s/cnngram/models/easy/mappings.pkl' % BASE_PATH)

    max_seq_len = 200
    word_emb_weight = np.zeros((len(dico_words), parameters['word_dim']))
    n_words = len(dico_words)

    #print '------gramcnn model----'
    print ' [*] Loading GRAMCNN tensorflow model (3min)...'
    gramcnn = GRAMCNN(n_words, len(char_to_id), len(pt_to_id),
                        use_word = parameters['use_word'],
                        use_char = parameters['use_char'],
                        use_pts = parameters['pts'],
                        num_classes = len(tag_to_id),
                        word_emb = parameters['word_dim'],
                        drop_out = 0,
                        word2vec = word_emb_weight,feature_maps=parameters['num_kernels'],#,200,200, 200,200],
                        kernels=parameters['kernels'], hidden_size = parameters['word_lstm_dim'], hidden_layers = parameters['hidden_layer'],
                        padding = parameters['padding'], max_seq_len = max_seq_len)



    #print '------gramcnn load----'
    gramcnn.load(models_path ,model_name)
    compilation = [opts, id_to_tag, word_to_id, char_to_id, tag_to_id, pt_to_id, lower, max_seq_len]

    print ' [*] Finished loading.'
    return compilation, parameters, gramcnn

def process_results(result_arr, sentence):

    _types = result_arr[0]
    _tokens = nltk.word_tokenize(sentence)
    result_dict = []

    for _type, _token in zip(_types, _tokens):
        _dict = {'entity': _token, 'type': str(_type)}
        result_dict.append(_dict)

    return result_dict

compilation, parameters, gramcnn = load_gramcnn()
opts, id_to_tag, word_to_id, char_to_id, tag_to_id, pt_to_id, lower, max_seq_len = compilation
# sentence = 'Number of glucocorticoid receptors in lymphocytes and their sensitivity to hormone action.'
# print evaluate_sentence(78, sentence)
