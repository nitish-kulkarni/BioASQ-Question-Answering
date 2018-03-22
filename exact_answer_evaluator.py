import re

class ExactAnswerEvaluator:

    def __init__(self):
        self.factoid_results = {}

    def eval(self, data):
        return self._eval_factoid_type(data)

    def _eval_factoid_type(self, data):
        questions = [question for question in data.questions if question.type == 'factoid']
        # ToDo: Fix this
        questions = [question for question in questions if question.exact_answer]
        N = len(questions)
        exact_match_count, soft_match_count = 0.0, 0.0
        total_mrr_exact, total_mrr_soft = 0.0, 0.0

        for question in questions:
            answers = question.exact_answer
            ref = question.exact_answer_ref
            exact_match, soft_match = False, False
            mrr_exact, mrr_soft = 0, 0
            # print ref, answers, question.qid
            for true_answer in ref:
                # true_answer = _normalize(true_answer)
                if true_answer in answers:
                    exact_match = True
                    mrr_exact = max(mrr_exact, 1.0 / (1 + answers.index(true_answer)))

                soft_matches = _soft_matches(true_answer, answers)
                if len(soft_matches) > 0:
                    soft_match = True
                    mrr_soft = max(mrr_soft, 1.0 / (1 + min(soft_matches)))

            exact_match_count += exact_match
            soft_match_count += soft_match
            total_mrr_exact += mrr_exact
            total_mrr_soft += mrr_soft

        self.factoid_results = {
            'Exact Matches': exact_match_count / N,
            'Soft Matches': soft_match_count / N,
            'MRR Exact': total_mrr_exact / N,
            'MRR Soft': total_mrr_soft / N
        }
        
        return self.factoid_results

def _soft_matches(ref, answers):
    answers = list(map(_normalize, answers))
    ref = _normalize(ref)
    matches = []
    for i, answer in enumerate(answers):
        if answer in ref or ref in answer:
            matches.append(i)
    return matches

def _normalize(string):
    return re.sub(r'\W+', '', string).lower()
