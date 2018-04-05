from svm.rank import SVMRank
import json

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

def get_data(file_name):

	file = open(file_name)
	data = json.load(file)['questions']

	return data

def preprocess_data(data):

	prep = []
	for question in data:
		obj = {}
		obj['answer'] = question['ideal_answer']
		if 'snippets' in question:
			obj['sentences'] = preprocess_sentences(get_sentences(question['snippets']))
		obj['type'] = question['type']
		obj['query'] = unicode(question['body']).encode("ascii", "ignore")
		obj['ner_entities'] = question['ner_entities'][:5]
		if len(obj['ner_entities'])>0:
			#if obj['type'] == 'factoid':  # TODO: uncomment
			prep.append(obj)

	return prep

def get_ground_truth(ner_entities, query):

	return [i+1 for i in range(len(ner_entities))]

def featurizer(ner_entities, query):

	return [[1.0, 3.0, 1.0] for i in range(len(ner_entities))], ner_entities

def get_features(ner_entities, query, i):

	# TODO : work with features
	X, ner_considered = featurizer(ner_entities, query)
	y = get_ground_truth(ner_considered, query)
	bundle = [i for k in range(len(ner_considered))]

	return X, y, bundle

def main():

	ranker = SVMRank()
	file_name = 'data/ners_BioASQ-trainingDataset6b.json'
	data = preprocess_data(get_data(file_name))

	for i, query in enumerate(data):
		ner_entities = query['ner_entities']
		X, y, bundle = get_features(ner_entities, query, i)
		ranker.feed(X, y, bundle)

	ranker.train_from_feed()
	ranker.save('weights')
		
if __name__ == "__main__":
	main()
