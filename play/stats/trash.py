		"""
		if type(answers) != type(list()):
			answers = [query['answer']]
		else:
			answers = answers[0]

		count_sentences_with_answer = 0

		print type(answers)
		"""

		"""
		for sentence in sentences:
			for answer in answers:
				if answer in sentence:
					count_sentences_with_answer += 1
					break

		print count_sentences_with_answer
		"""
		


	"""
	sent_sizes = []
	for query in factoid_queries:
		#answers = 
		#sent_avg += query
		sentences = query['sentences']
		if len(sentences)<1:
			print query['answer']
		#sent_sizes.append(len(sentences))

	#sent_avg = np.array(sent_sizes).mean()
	#print sent_avg
	"""

['acrokeratosis', 'bazex', 'paraneoplastica', 'syndrome', 'carcinoma', 'cell', 'tract', 'case', 'treatment', 'dermatosis']
[u'bazex', u'syndrome']



['orteronel', 'prostate', 'cancer', 'patients', 'treatment', 'androgen', 'tak-700', 'phase', 'agents', 'inhibitor']
[u'prostate', u'cancer']



['pannexin1', 'channels', 'atp', 'release', 'membrane', 'panx1', 'channel', 'junction', 'cell', 'connexins']
[u'plasma', u'membrane']



['inheritance', 'disease', 'copper', 'wilson', 'gene', 'disorder', 'wd', 'patients', 'mode', 'brain']
[u'recessive']



['inheritance', 'fshd', 'dystrophy', 'facioscapulohumeral', 'weakness', 'disorder', 'family', 'involvement', 'pattern', 'cases']
[u'dominant']



['hilic', 'interaction', 'method', 'hydrophilic-interaction', 'chromatography', 'lc-ms/ms', 'mass', 'spectrometry', 'chromatographic', 'hydrophilic']
[u'interaction', u'chromatography', u'hydrophilic']



['bromodomain', 'structure', 'fold', 'brg1', 'bromodomains', 'structures', 'za', 'bundle', 'loop', 'addition']
[u'fold']



['substrates', 'amine', 'isotopic', 'labeling', 'tails', 'protease', 'n-termini', 'analysis', 'protein', 'discovery']
[u'substrates', u'isotopic', u'tails', u'amine', u'labeling']



['fusion', 'ewing', 'sarcoma', 'protein', 'tumors', 'genes', 'development', 'transcription', 'ews-fli1', 'target']
[u'ews/fli1']



['treatment', 'netherlands', 'clean', 'mr', 'stroke', 'trial', 'trials', 'multicenter', 'heart', 'protocol']
[u'stroke']