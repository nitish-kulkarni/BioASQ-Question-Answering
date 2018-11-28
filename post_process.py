import os
import sys
import json
import pdb
from collections import Counter
def read_json(filename):
	with open(filename) as f:
		a=json.load(f)
	return a

def count_key(key_type,object):
	a=[object[0]['type'] for k in object if k.get('type')==key_type]
	#print(a)
	return len(a)

def get_val_key(keyname,keyval,object):
	keyname=str(keyname)
	print(keyname)
	a=[object for k in object if k.get(keyname)==keyval]
	return a

def compute_acc_type(inp_list,output_list):
	for question in inp_list:
		print(question.keys())
		exact_answer=question['exact_answer']
		print(exact_answer)
									

def compute_accuracy(input_json,output_json):
	input_question_json=read_json(input_json)
	output_answer_json=read_json(output_json)
	inp_lst_count=count_key('list',input_question_json['questions'])
	print('list type is',inp_lst_count)
	inp_factoid_count=count_key('factoid',input_question_json['questions'])
	print('factoid type is',inp_factoid_count)
	inp_yesno_count=count_key('yesno',input_question_json['questions'])
	print('yesno type is',inp_yesno_count)
	op_lst_count=count_key('list',output_answer_json['questions'])
	print('list type is',op_lst_count)
	op_factoid_count=count_key('factoid',output_answer_json['questions'])
	print('factoid type is',op_factoid_count)
	op_yesno_count=count_key('yesno',output_answer_json['questions'])
	print('yesno type is',op_yesno_count)
	op_summary_count=count_key('summary',output_answer_json['questions'])
	print('summary type is',op_summary_count)
	#print(input_question_json['questions'][0])
	#print(input_question_json['questions'][0].keys())
	
	output_list_type=get_val_key('type','list',output_answer_json['questions'])
	print(len(output_list_type))
	print(output_list_type[0])
	print(output_list_type[1])		
	input_question_list=[]
	pdb.set_trace()
	for question in output_list_type:
		print(question['id'])
		#print(question['id'])
		#print(get_val_key('id',question['id'],input_question_json['questions']))
		print(question['body'])
		input_question_list.append(get_val_key('body',question['body'],input_question_json['questions'])[0])
	
	print(len(input_question_list))	
	acc=compute_acc_type(input_question_list,output_list_type)	
	
	

if __name__=="__main__":
	compute_accuracy(sys.argv[1],sys.argv[2])


