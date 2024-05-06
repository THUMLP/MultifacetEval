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
        return None
    for key in options:
        if key in response_str:
            response_str = response_str.replace(key, options[key])
        if key.lower() in response_str:
            response_str = response_str.replace(key.lower(), options[key])
        if key[0].lower()+key[1:] in response_str:
            response_str = response_str.replace(key[0].lower()+key[1:], options[key])
    pattern=[
        r"^\s*Option\s*([A-E])",
        r"^\s*([A-E])\s*\Z",
        r"^\s*([A-E])\s+",
        r"^\s*([A-E])(?:,|:|\n|\)|\.)",
        r"^\s*\"([A-E])\"",
        r"Answers?:\s*([A-E])",
        r"(?:The|the) (?:correct )?answer (?:for .*? )?is:?\s*(?:option )?([A-E])",
        r"^Option ([A-E]) is the (?:correct )answer",
        r"Correct Answer:\s*([A-E])",
        r"Therefore, the answer (?:are|is|would be|should be)\s*(?: a)?:?\s*\**\"?(?:option)?\s*([A-E])",
        r"(?:Therefore, )?(?:The|the)[^\.]*(?:are|is|would be|should be)\s*(?: a)?:?\s*\**\"?(?:option)?\s*([A-E])",
        r"(?:The|the)[^\.]*(?:most|best) [^\.]* (?:is|would be|are|should be)\s*:?(?:an|the)?\s*([A-E])",
        # r"answer is therefore: ([A-E])",
        # r"^Options?:? ([A-E]) (?:are|is) the correct answers?",
        r"The answer is that options? ([A-E])",
        r"(?:Option )?([A-E])[^\.]*(?:most|best)[^\.]*\.",
        r"(?:are|is) consistent with ([A-E])",
        
    ]
    ans_list=[]
    response_str = response_str.strip()
    if len(response_str) == 0:
        return None
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
        r"Therefore, the (?:correct )?answer is:?\s*(incorrect|not correct|false|wrong|correct|consistent|true|yes|no)",
        r"(incorrect|not correct|false|wrong)",
        r"not the (?:best|most appropriate)"
        # r"(correct|consistent|true|yes)(?! answer)",
        # r"(no)\.?\s*"
        # r"^\s*([A-E])\s+",
        
    ]
    ans_list = []
    # hit_pattern = ''
    hit_text = ''
    for p in pattern:
        if len(ans_list)==0:
            ans_list=re.findall(p,response_str,re.I|re.DOTALL)
            if len(ans_list) > 0:
                hit_text = re.search(p,response_str,re.I|re.DOTALL).group(0)
        else:
            break
    if len(ans_list) == 0:
        # if 'correct' in response_str.lower():
        #     print('w')
        return None
    answer = ans_list[0]
    for one in ['incorrect','not correct','false','no','wrong','not the best','not the most appropriate']:
        if one in answer.lower():
            return 'F'
    for one in ['true','correct','consistent','yes']:
        if one in answer.lower():
            return 'T'
    
    # if not response_str.startswith('Question') and len(response_str.strip().split())<=10:
    #     print(response_str)
    return None

def extract_ans(response_str,options, alice_ans):
    tf_pred = extract_judge_ans(response_str)
    option_pred = extract_option_ans(response_str,options)
    # if tf_pred == None or (tf_pred == 'F' and option_pred == None):
    #     if tf_pred == None:
    #         # print('type-I:\t'+response_str)
    #         pass
            
    #     else:        
    #         print('type-II:\t'+response_str)
    #         if 'the most likely' in response_str:
    #             print('y')
    if tf_pred == 'T':
        option_pred = alice_ans

    return [tf_pred, option_pred]

def answer_vote(answers, alice_ans):
    tf_cnt = [0,0]
    op_cnt = [0,0,0,0,0]
    
    for one in answers:
        if one[0] == 'T':
            tf_cnt[0] += 1
        elif one[0] == 'F':
            tf_cnt[1] += 1
        if one[1] != None:
            op_cnt[ord(one[1])-ord('A')] += 1
    if sum(tf_cnt) == 0:
        tf_ans = None
    else:
        if tf_cnt[0] > tf_cnt[1]:
            tf_ans = 'T'
        elif tf_cnt[0] < tf_cnt[1]:
            tf_ans = 'F'
        else:
            tf_ans = random.choice(['T','F'])
    if tf_ans == 'T':
        op_ans = alice_ans
    else:
        
        if sum(op_cnt) == 0:
            op_ans = None
        else:
            pos_pred_candidates = [i for i in range(5) if op_cnt[i] == max(op_cnt)]
            pos_pred = random.choice(pos_pred_candidates)
            op_ans = chr(pos_pred+ord('A'))
    return [tf_ans, op_ans]

def majority_vote(answers):
    answer_count = {}
    for answer in answers:
        ans_text = str(answer[0])+'_'+str(answer[1])
        if ans_text not in answer_count:
            answer_count[ans_text] = 0
        answer_count[ans_text] += 1
    max_vote = -1
    max_cands = []
    for key in answer_count:
        if answer_count[key] > max_vote:
            max_vote = answer_count[key]
    for key in answer_count:
        if answer_count[key] == max_vote:
            max_cands += [key]
    answer = random.choice(max_cands)
    answer = answer.split('_')
    new_answer = []
    for one in answer:
        if one == 'None':
            new_answer.append(None)
        else:
            new_answer.append(one)
    return new_answer
    
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
    for n_idx, name in enumerate(args.models_been_eval):
        full_name = os.path.join(result_dir, name+'_RQ_results_cot.json')
        out_full_name = os.path.join(result_dir, name+'_RQ_results_processed_cot.json')
        outf = open(out_full_name,'w')
        model_name = name
        fail, ttl, corr, hit_corr = 0,0,0,0
        with open(full_name, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                T_ques = entry[1]
                F_ques = entry[2]
                direction = 'forward'
                options = re.findall(r"([A-E]): (.*?)(?:\t|\n|\Z|\.)",T_ques)
                options = {v[1]:v[0] for v in options}
                T_ans = entry[3]
                F_ans = entry[4]
                T_replys = entry[-2]
                F_replys = entry[-1]
                T_preds = [extract_ans(T_reply,options,re.search(r"Alice\'s answer: ([A-E])",T_ques)[1]) for T_reply in T_replys]
                F_preds = [extract_ans(F_reply,options,re.search(r"Alice\'s answer: ([A-E])",F_ques)[1]) for F_reply in F_replys]
                T_pred, F_pred = answer_vote(T_preds,re.search(r"Alice\'s answer: ([A-E])",T_ques)[1]), answer_vote(F_preds,re.search(r"Alice\'s answer: ([A-E])",F_ques)[1])
                # T_pred, F_pred = majority_vote(T_preds), majority_vote(F_preds)
                T_alice, F_alice = re.search(r"Alice\'s answer: ([A-E])",T_ques)[1],re.search(r"Alice\'s answer: ([A-E])",F_ques)[1]
                T_pred, miss = missing_value_padding(T_pred,T_alice)
                F_pred, miss = missing_value_padding(F_pred,F_alice)
                entry += [T_pred, F_pred]
                outf.write(json.dumps(entry,ensure_ascii=False)+'\n')
        all_ttl, all_fail, all_corr, all_tf_corr, all_op_corr = 0,0,0,0,0
        outf.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_been_eval", type=list, default=['chatglm-6B','vicuna_7B','llama2-7B','bloomz-7b1-mt','meditron-7B','pulse-7B','vicuna_13B','llama2-70B', 'meditron-70B', 'clinicalcamel-70B', 'med42-70B','gemini-pro','gpt-3.5-turbo'])    
    args = parser.parse_args()
    main(args)