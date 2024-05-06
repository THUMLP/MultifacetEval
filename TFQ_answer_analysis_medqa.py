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
    for one in ['incorrect','not correct','false','False','No','wrong']:
        if one in response_str:
            return 'F'
    for one in ['true','correct','consistent','True','Yes']:
        if one in response_str:
            return 'T'
    
    # if not response_str.startswith('Question') and len(response_str.strip().split())<=10:
    #     print(response_str)
    return None
def main(args):
    random.seed(48)
    result_dir = 'results/medqa/TFQ'
    # names = ['med42-70B']

    for name in args.models_been_eval:
        full_name = os.path.join(result_dir, name+'_TFQ_results.json')
        out_full_name = os.path.join(result_dir, name+'_TFQ_results_processed.json')
        outf = open(out_full_name,'w')
        inconsistent = 0
        with open(full_name, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                pos_T_reply = entry[9]
                neg_T_reply = entry[10]
                pos_F_reply = entry[11]
                neg_F_reply = entry[12]
                pos_T_pred = extract_ans(pos_T_reply)
                neg_T_pred = extract_ans(neg_T_reply)
                pos_F_pred = extract_ans(pos_F_reply)
                neg_F_pred = extract_ans(neg_F_reply)
                if not pos_T_pred:
                    # print(reply)
                    pos_T_pred = random.choice(['T','F'])

                if not neg_T_pred:
                    # print(reply)
                    neg_T_pred = random.choice(['T','F'])
                if pos_T_pred == neg_T_pred:
                    inconsistent += 1

                if not pos_F_pred:
                    # print(reply)
                    pos_F_pred = random.choice(['T','F'])

                if not neg_F_pred:
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