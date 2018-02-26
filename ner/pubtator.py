"""Extract bio concepts from pubtator
"""

import json
import requests

CACHE = {}

def load_from_uri(uri):
    r = requests.get(uri)
    r.encoding = 'utf-8'

    return r.text.strip('\n')

def get_bio_concepts(doc_id, doc_text, concept='all'):
    if concept == 'all':
        concept = 'BioConcept'

    key = (concept, doc_id)
    if key in CACHE:
        return CACHE[key]

    uri = 'https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/%s/%d/JSON/' % (concept, doc_id)
    raw_data = load_from_uri(uri)
    
    if raw_data.startswith('[Error]'):
        return {}
    
    data = json.loads(raw_data)
    bioconcepts = _bioconcepts_from_pubtator_data(data)

    CACHE[key] = bioconcepts
    return bioconcepts

def get_bio_concepts_multiple(doc_ids, doc_texts, concept='all'):
    if len(doc_ids) == 0:
        return []

    if len(doc_ids) == 1:
        return get_bio_concepts(doc_ids[0], doc_texts[0], concept=concept)

    if concept == 'all':
        concept = 'BioConcept'

    key = (concept, tuple(doc_ids))
    if key in CACHE:
        return CACHE[key]

    uri = 'https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/%s/%s/JSON/' % (concept, ','.join(list(map(str, doc_ids))))
    raw_data = load_from_uri(uri)
    
    if raw_data.startswith('[Error]'):
        return [{} for _ in range(len(doc_ids))]
    
    data = json.loads('[%s]' % raw_data[1:-1])
    bioconcepts = []
    for id_data in data:
        bioconcepts += _bioconcepts_from_pubtator_data(id_data)

    CACHE[key] = bioconcepts
    return bioconcepts

def _bioconcepts_from_pubtator_data(data):
    bioconcepts = []
    for denotation in data['denotations']:
        bioconcepts.append({
            'entity': data['text'][int(denotation['span']['begin']):int(int(denotation['span']['end']))],
            'type': denotation['obj'].split(':')[0]
        })

    return bioconcepts

def get_doctext_from_docid(doc_id):
    uri = 'https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/none/%d/JSON/' % doc_id
    raw_data = load_from_uri(uri)
    if raw_data.startswith('[Error]'):
        return ''

    return raw_data['text']
