#!/usr/bin/python

import sys, os, subprocess, json
def rouge(gold_results, system_results):
        os.environ['ROUGE_EVAL_HOME']="./data"
        #os.environ['PERL5LIB']="./libs/perl5/lib/perl5"
		#os.environ['PYTHONPATH']='./libs/python2.7/site-packages/'
        d=json.loads(open(system_results).read())
        g=json.loads(open(gold_results).read())
        i=1
        string='<ROUGE-EVAL version="1.0">\n'
        for question in d['questions']:
                string+='<EVAL ID="{0}">\n<PEER-ROOT>\n./bioasq\n</PEER-ROOT>\n<MODEL-ROOT>\n./bioasq\n</MODEL-ROOT>\n<INPUT-FORMAT TYPE="SEE">\n</INPUT-FORMAT>\n<PEERS>\n<P ID="1">system{0}.html</P>\n</PEERS>\n<MODELS>\n'.format(i)
		gquestion=[qu for qu in g['questions'] if qu['id'] == question['id']][0]
		for j in range(len(gquestion['ideal_answer'])):
			string += '<M ID="{0}">golden{1}.{2}.html</M>\n'.format(j,i,j)
		string +=   '</MODELS>\n</EVAL>\n' #check len of golden-adapt
                string_system='<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</body>\n</html>'.format(question['ideal_answer'].encode('ascii', 'ignore'))#This is the participant answer
                pt="./bioasq/system%d.html" %i
                with open(pt, 'w') as out:
			os.chmod(pt, 0o777)
                        out.write(string_system)
		for j in range(len(gquestion['ideal_answer'])):
			string_golden='<html>\n<head>\n<title>golden</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</body>\n</html>'.format(gquestion['ideal_answer'][j].encode('ascii','ignore'))
			pt="./bioasq/golden%d.%d.html" %(i,j)
			with open(pt, 'w') as out:
        	                os.chmod(pt, 0o777)
                	        out.write(string_golden)
	
                i+=1
        string+="</ROUGE-EVAL>"
        pt="./bioasq/bioasq-test.xml"
        with open(pt, 'w') as out:
		os.chmod(pt, 0o777)
                out.write(string)
        i=1
        command="perl ./ROUGE-1.5.5.pl -c 95 -2 4 -u -x -n 4 ./bioasq/bioasq-test.xml 1"
        result=subprocess.Popen(command.split(),  stdout=subprocess.PIPE)
        p=result.communicate()
	a=[float(x) for x in p[0].split()]
        #os.remove("/var/www/participants_area/bioasq-test.xml")
       	path_to_delete="./bioasq"
        for f in os.listdir(path_to_delete):
               os.remove(os.path.join(path_to_delete,f))
	print a
        return a
rouge(sys.argv[1], sys.argv[2])

