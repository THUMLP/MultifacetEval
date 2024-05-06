'''
    Extract answer choice from the plain text with the re package
'''
import re
import os
import json
import random
import pandas as pd
import argparse
def extract_option_ans(response_str, options):
    if len(response_str) == 0:
        return []
    for key in options:
        if key in response_str:
            response_str = response_str.replace(key, options[key])
    pattern=[
        r"^\s*Option\s*([A-E])",
        r"^\s*([A-E])\s*\Z",
        r"^\s*([A-E])\s+",
        r"^\s*([A-E])(?:,|:|\n|\)|\.)",
        r"^\s*\"([A-E])\"",
        r"Answers?:\s*([A-E])",
        r"(?:The|the) (?:correct )?answer (?:for .*? )?is:?\s*(?:option )?([A-E])",
        r"^Option ([A-E]) is the (?:correct )answer",
        r"Correct Answer:\s*([A-E])"
        
    ]
    ans_list=[]
    response_str = response_str.strip()
    if len(response_str) == 0:
        return []
    # if response_str[0] in ['A','B','C','D','E']:
    #     ans_list.append(response_str[0])
    for p in pattern:
        if len(ans_list)==0:
            ans_list=re.findall(p,response_str,re.DOTALL)
        else:
            break
    # if len(ans_list) == 0:
    #     print(response_str)
    if len(ans_list)>0:
        ans_list  = ans_list[0]
    else:
        ans_list = None
    return ans_list

def extract_judge_ans(response_str):
    if len(response_str) == 0:
        return None
    # if response_str == 'false':
    #     return 'F'
    # if response_str == '是的':
    #     return 'T'
    # for one in ['wrong']:
    #     if response_str.startswith(one):
    #         return 'F'
    # for one in ['正确']:
    #     if response_str.startswith(one):
    #         return 'T'
    pattern=[
        r"(incorrect|not correct|false|wrong)",
        r"(correct|consistent|true|yes)(?! answer)",
        r"(no)\.?\s*"
        r"^\s*([A-E])\s+",
        
    ]
    ans_list = []
    for p in pattern:
        if len(ans_list)==0:
            ans_list=re.findall(p,response_str,re.I|re.DOTALL)
        else:
            break
    if len(ans_list) == 0:
        # if 'correct' in response_str.lower():
        #     print('w')
        return None
    answer = ans_list[0]
    for one in ['incorrect','not correct','false','no','wrong']:
        if one in answer.lower():
            return 'F'
    for one in ['true','correct','consistent','yes']:
        if one in answer.lower():
            return 'T'
    
    # if not response_str.startswith('Question') and len(response_str.strip().split())<=10:
    #     print(response_str)
    return None

def extract_ans(response_str,options):
    tf_pred = extract_judge_ans(response_str)
    option_pred = extract_option_ans(response_str,options)
    # if tf_pred == None or (tf_pred == 'F' and option_pred == None):
    #     if tf_pred == None:
    #         print('type-I:\t'+response_str)
            
    #     else:
    #         print('type-II:\t'+response_str)
    return [tf_pred, option_pred]

def missing_value_padding(answer, alice_ans):
    # 1. check the verify result:
    miss = 1 if (answer[0] == None or (answer[0] == 'F' and answer[1] == None)) else 0
    if answer[0] is not None and answer[1] is not None:
        return answer,miss
    
    if answer[0] == None:
        answer[0] = random.choice(['T','F'])

    if answer[1] == None:
        if answer[0] == 'T':
            answer[1] = alice_ans
        else:
            answer[1] = random.choice(list({'A','B','C','D','E'}-{alice_ans}))
    return answer,miss  
def main(args):
    random.seed(48)
    result_dir = 'results/medqa/RQ'
    # names = ['gemini-pro']
    # names = ['chatglm-6B','meditron-7B','pulse-7B','vicuna_7B','llama2-7B','bloomz-7b1-mt','vicuna_13B','med42-70B','llama2-70B','meditron-70B','clinicalcamel-70B','gemini-pro','gpt-3.5-turbo']

    # outff = open('wrong_ans.txt','w')

        
        
    for n_idx, name in enumerate(args.models_been_eval):
        full_name = os.path.join(result_dir, name+'_RQ_results.json')
        out_full_name = os.path.join(result_dir, name+'_RQ_results_processed.json')
        outf = open(out_full_name,'w')
        model_name = name
        fail, ttl, corr, hit_corr = 0,0,0,0
        with open(full_name, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                T_ques = entry[1]
                F_ques = entry[2]
                direction = 'forward'
                options = re.findall(r"([A-E]): (.*?)(?:\t|\Z|\.)",T_ques)
                options = {v[1]:v[0] for v in options}
                T_ans = entry[3]
                F_ans = entry[4]
                T_reply = entry[-2]
                F_reply = entry[-1]
                T_pred = extract_ans(T_reply,options)
                F_pred = extract_ans(F_reply,options)
                T_alice, F_alice = re.search(r"Alice\'s answer: ([A-E])",T_ques)[1],re.search(r"Alice\'s answer: ([A-E])",F_ques)[1]
                #Missing value preprocessing
                
                # if T_pred[0] == None or (T_pred[0] == 'F' and T_pred[1] == None):
                #     results[direction]['T'][0] += 1
                #     if T_pred[0] == None:
                #         T_pred[0] = random.choice(['T','F'])
                #     if T_pred[0] == 'T':
                #         T_pred[1] = T_alice
                #     elif T_pred[1] == None:
                #         T_pred[1] = random.choice(list({'A','B','C','D','E'}-{T_alice}))
                # else:
                #     if T_pred[0] == 'T':
                #         T_pred[1] = T_ans[1]
                T_pred, miss = missing_value_padding(T_pred,T_alice)

                # if F_pred[0] == None or (F_pred[0] == 'F' and F_pred[1] == None):
                #     results[direction]['F'][0] += 1
                #     if F_pred[0] == None:
                #         F_pred[0] = random.choice(['T','F'])
                #     if F_pred[0] == 'T':
                #         F_pred[1] = F_alice
                #     elif F_pred[1] == None:
                #         F_pred[1] = random.choice(list({'A','B','C','D','E'}-{F_alice}))
                # else:
                #     if F_pred[0] == 'T':
                #         F_pred[1] = re.search(r"Alice\'s answer: ([A-E])",F_ques)[1]
                F_pred,miss = missing_value_padding(F_pred,F_alice)
                # if pos_F_pred == neg_F_pred:
                #     inconsistent += 1
                entry += [T_pred, F_pred]
                outf.write(json.dumps(entry,ensure_ascii=False)+'\n')
        all_ttl, all_fail, all_corr, all_tf_corr, all_op_corr = 0,0,0,0,0
        outf.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_been_eval", type=list, default=['chatglm-6B','vicuna_7B','llama2-7B','bloomz-7b1-mt','meditron-7B','pulse-7B','vicuna_13B','llama2-70B', 'meditron-70B', 'clinicalcamel-70B', 'med42-70B','gemini-pro','gpt-3.5-turbo'])    
    args = parser.parse_args()
    main(args)