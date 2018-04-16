import cPickle as pickle
import numpy as np

def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def main():

    questions = load_object('pkl/questions.pkl')
    questions = questions
    percentual = []

    for i, question in enumerate(questions):
        query = question['query']
        answers = question['answers']
        candidates = [a['candidate'] for a in question['candidates']]
        is_present = False

        for candidate in candidates:
            for answer in answers:
                if answer.lower() == candidate.lower():
                    is_present = True

        if is_present:
            percentual.append(1)
        else:
            percentual.append(0)
            print query, i
            print '\n'
            print answers
            print '\n'
            print '\n\n\n'

    percent = np.array(percentual).mean()
    print percent

    num = 391
    ref = 'chediak higashi syndrome'

    print '\n\n'
    print questions[num]['query']
    print questions[num]['answers']
    print [c['candidate'] for c in questions[num]['candidates'] if ref in c['candidate']]

    """
    percentual = []
    for i, question in enumerate(questions):
        answers = question['answers']
        candidates = [a['candidate'] for a in question['candidates']]
        is_present = False
        
        for candidate in candidates[:5]:
            for answer in answers:
                if answer.lower() == candidate.lower():
                    is_present = True

        if is_present:
            percentual.append(1)
        else:
            percentual.append(0)

    percent = np.array(percentual).mean()
    """


if __name__ == '__main__':
    main()