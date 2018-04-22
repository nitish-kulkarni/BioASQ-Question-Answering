"""
Computes the BLEU, ROUGE
using the COCO metrics scripts
"""
from .bleu.bleu import Bleu
from .rouge.rouge import Rouge
import glob


def load_textfiles(references, hypothesis):

    """
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypothesis)}
    # take out newlines before creating dictionary
    raw_refs = [map(str.strip, r) for r in zip(references)]
    refs = {idx: rr for idx, rr in enumerate(raw_refs)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError("There is a sentence number mismatch between the inputs")

    print(refs)
    """

    refs = {0: references}
    hypo = {0: hypothesis}


    return refs, hypo


def __score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1"]),
        (Rouge(), "ROUGE_L"),
    ]

    #print('---')
    #print(ref)


    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores

def get_score(str_reference, str_hypothesis):

	reference = [str_reference]
	hypothesis =  [str_hypothesis]
	ref, hypo = load_textfiles(reference, hypothesis)
	score_map = __score(ref, hypo)

	return score_map


def test():
    # Feed in the directory where the hypothesis summary and true summary is stored
    hyp_file = glob.glob('hypothesis/*')
    ref_file = glob.glob('reference/*')

    BLEU_1 = 0.
    BLEU_2 = 0.
    BLEU_3 = 0.
    BLEU_4 = 0.
    ROUGE_L = 0.
    num_files = 0
    for reference_file, hypothesis_file in zip(ref_file, hyp_file):
        num_files += 1
        #print reference_file, hypothesis_file

        with open(reference_file) as rf:
            reference = rf.readlines()

        with open(hypothesis_file) as hf:
            hypothesis = hf.readlines()

        reference = str(reference[0].strip())
        hypothesis =  str(hypothesis[0].strip())

        score_map = get_score(reference, hypothesis)
        #print score_map
