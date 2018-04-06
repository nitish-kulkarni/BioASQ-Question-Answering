"""Generate sentences from questions
"""

from bllipparser import RerankingParser
from nltk.tree import Tree
from tqdm import tqdm

def pos_tags(q, rrp):
    s = rrp.simple_parse(q.encode("ascii", 'ignore'))
    tree = Tree.fromstring(s)
    return tree.pos()

def q2s(q, rrp):
    postags = pos_tags(q, rrp)
    words = []
    first = postags[0][0].lower()
    inserted = False
    for word, pt in postags[1:]:
        if word == '?':
            words.append('.')
            continue
        if inserted:
            words.append(word)
            continue
        else:
            if pt.startswith('VB'):
                words.append(first)
                words.append(word)
                inserted = True
            elif pt in ['EX']:
                words.append(word)
                words.append(first)
                inserted = True
            else:
                words.append(word)
    return ' '.join(words)

def set_assertions_for_yesno_questions(data):
    rrp = RerankingParser.fetch_and_load('GENIA+PubMed', verbose=True)
    yesno = data.get_questions_of_type('yesno')
    for q in tqdm(yesno):
        q.assertion_pos = q2s(q.question, rrp)
