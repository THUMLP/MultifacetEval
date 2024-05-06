'''
    Extract answer choice from the plain text with the re package
'''
import re
import os
import json
import random
import pandas as pd
import argparse

def extract_ans(response_str):
    if len(response_str) == 0:
        return False
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
    for one in ['incorrect','not correct','false','False','No','wrong']:
        if one in response_str:
            return 'F'
    for one in ['true','correct','consistent','True','Yes']:
        if one in response_str:
            return 'T'
    
    # if not response_str.startswith('Question') and len(response_str.strip().split())<=10:
    #     print(response_str)
    return False

def answer_vote(answers):
    ans_cnt = [0,0]
    for one in answers:
        if one == 'T':
            ans_cnt[0] += 1
        else:
            ans_cnt[1] += 1
    if ans_cnt[0] > ans_cnt[1]:
        ans = 'T'
    elif ans_cnt[0] < ans_cnt[1]:
        ans = 'F'
    else:
        ans = random.choice(['T','F'])
    if sum(ans_cnt) == 0:
        return None
    else:
        return ans
def main(args):
    random.seed(48)
    result_dir = 'results/medqa/TFQ'
    # names = ['med42-70B']
    # names = ['chatglm-6B','meditron-7B','pulse-7B','vicuna_7B','llama2-7B','bloomz-7b1-mt','vicuna_13B','med42-70B','llama2-70B','meditron-70B','clinicalcamel-70B','gemini-pro','gpt-3.5-turbo']

    for name in args.models_been_eval:
        full_name = os.path.join(result_dir, name+'_TFQ_results_cot.json')
        out_full_name = os.path.join(result_dir, name+'_TFQ_results_processed_cot.json')
        outf = open(out_full_name,'w')
        inconsistent = 0
        with open(full_name, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                pos_T_replys = entry[9]
                neg_T_replys = entry[10]
                pos_F_replys = entry[11]
                neg_F_replys = entry[12]
                pos_T_preds = [extract_ans(pos_T_reply) for pos_T_reply in pos_T_replys]
                neg_T_preds = [extract_ans(neg_T_reply) for neg_T_reply in neg_T_replys]
                pos_F_preds = [extract_ans(pos_F_reply) for pos_F_reply in pos_F_replys]
                neg_F_preds = [extract_ans(neg_F_reply) for neg_F_reply in neg_F_replys]
                pos_T_pred, neg_T_pred, pos_F_pred, neg_F_pred = answer_vote(pos_T_preds), answer_vote(neg_T_preds), answer_vote(pos_F_preds), answer_vote(neg_F_preds)
                if not pos_T_pred:
                    # print(reply)
                    pos_T_pred = random.choice(['T','F'])

                if not neg_T_pred:
                    # outff.write('Ans: {}\n'.format(neg_T_ques,neg_T_reply))
                    # print(reply)
                    neg_T_pred = random.choice(['T','F'])
                if pos_T_pred == neg_T_pred:
                    inconsistent += 1

                if not pos_F_pred:
                    # outff.write('Ans: {}\n'.format(pos_F_ques,pos_F_reply))
                    # print(reply)
                    pos_F_pred = random.choice(['T','F'])

                if not neg_F_pred:
                    # outff.write('Ans: {}\n'.format(neg_F_ques,neg_F_reply))
                    # print(reply)
                    neg_F_pred = random.choice(['T','F'])
                if pos_F_pred == neg_F_pred:
                    inconsistent += 1
                entry += [pos_T_pred, neg_T_pred, pos_F_pred, neg_F_pred]
                outf.write(json.dumps(entry,ensure_ascii=False)+'\n')
        outf.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_been_eval", type=list, default=['chatglm-6B','vicuna_7B','llama2-7B','bloomz-7b1-mt','meditron-7B','pulse-7B','vicuna_13B','llama2-70B', 'meditron-70B', 'clinicalcamel-70B', 'med42-70B','gemini-pro','gpt-3.5-turbo'])    
    args = parser.parse_args()
    main(args)