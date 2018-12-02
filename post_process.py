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
        a=[k for k in object if k.get(keyname)==keyval]
        return a

def list_flatten(inp_list):
    return [item for subitem in inp_list for item in subitem]

def compute_acc_type(inp_list,output_list):
        soft_measure=[]
        hard_measure=[]
        for question in inp_list:
                #print(question.keys())
                exact_answer=question['exact_answer']
                exact_answer_flatted=list_flatten(exact_answer)
                id_ques=question['id']
                output_answer=get_val_key('id',id_ques,output_list) 
                #print(output_answer)
                predicted_answer=output_answer[0]['ideal_answer']
                print(predicted_answer)
                print(exact_answer_flatted)
                if any(exact_ans_instance in predicted_answer for exact_ans_instance in exact_answer_flatted):
                    soft_measure.append(1)
                else:
                    soft_measure.append(0)
                if all(exact_ans_instance in predicted_answer for exact_ans_instance in exact_answer_flatted):
                    hard_measure.append(1)
                else:
                    hard_measure.append(0)
        print(soft_measure)
        print(hard_measure)
        return soft_measure.count(1)/float(len(soft_measure)), hard_measure.count(1)/float(len(hard_measure))
                #print(id_ques)
                #print(exact_answer)

def get_input_question_list(input_list_json,output_list):
        input_question_list=[]
        for question in output_list:
            input_question_list.append(get_val_key('id',question['id'],input_list_json['questions'])[0])
        #print(input_question_list)
        assert(len(output_list)==len(input_question_list))
        return input_question_list

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
        input_question_list=get_input_question_list(input_question_json,output_list_type)
        print(len(input_question_list))        
        soft_measure,hard_measure=compute_acc_type(input_question_list,output_list_type)      
        print("List type soft and hard measure", soft_measure,hard_measure)
        
        output_yesno_type=get_val_key('type','yesno',output_answer_json['questions'])
        #print(len(output_yesno_type))
        input_question_list=get_input_question_list(input_question_json,output_yesno_type)
        soft_measure,hard_measure=compute_acc_type(input_question_list,output_yesno_type)      
        print("Yesno type soft and hard measure", soft_measure,hard_measure)
       
        output_factoid_type=get_val_key('type','factoid',output_answer_json['questions'])
        input_question_list=get_input_question_list(input_question_json,output_factoid_type)
        soft_measure,hard_measure=compute_acc_type(input_question_list,output_factoid_type)      
        print("Factoid type soft and hard measure", soft_measure,hard_measure)


        
        

if __name__=="__main__":
        compute_accuracy(sys.argv[1],sys.argv[2])


