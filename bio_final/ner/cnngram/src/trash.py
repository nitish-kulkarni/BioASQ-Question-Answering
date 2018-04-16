#parameters, sess, raw_sentences, parsed_sentences,id_to_tag

def evaluate(parameters, sess, raw_sentences, parsed_sentences,
             id_to_tag, remove = True, padding = False, max_seq_len = 200, use_pts = False):
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)

    print "Preparing Data"
    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        inputs,s_len = create_input(data, parameters, add_label = False, singletons = None, padding = padding, max_seq_len = max_seq_len,
            use_pts = use_pts)
        
        # if parameters['crf']:
        #     y_preds = np.array(f_eval(*input))[1:-1]
        # else:
        #     y_preds = f_eval(*input).argmax(axis=1)
        #print inputs
        temp = []
        temp.append(s_len)
        y_preds = sess.test(inputs, temp)


def create_input(data, parameters, add_label, singletons=None, padding = False, max_seq_len = 200, use_pts = False):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    input = {}
    words = data['words']
    word_len = len(words)
    chars = data['chars']
    tags = data['tags']
    char_for = []
    max_length = 0
    if not padding:
        if singletons is not None:
            words = insert_singletons(words, singletons)
        if parameters['cap_dim']:
            caps = data['caps']
        char_for, char_rev, char_pos, max_length = pad_word_chars(chars, singletons)
        pts = data['pts']
    else:
        if singletons is not None:
            words = insert_singletons(words, singletons)
        words = padding_word(words, max_seq_len)
        caps = padding_word(data['caps'], max_seq_len)
        pts = padding_word(data['pts'], max_seq_len)
        tags = padding_word(tags, max_seq_len)
        char_for, char_rev, char_pos, max_length = pad_word_chars(chars, singletons)
        char_for = padding_chars(char_for, max_seq_len, max_length)
    if parameters['word_dim']:
        input['word'] = words
    if parameters['char_dim']:
        input['char_for'] = char_for
    if parameters['cap_dim']:
        input['cap'] = caps
    if add_label:
        input['label'] = tags
    if use_pts:
        input['pts'] = pts
    return input, word_len