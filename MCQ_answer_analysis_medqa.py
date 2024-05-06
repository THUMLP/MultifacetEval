'''
    Extract answer choice from the plain text with the re package
'''
import re
import os
import json
import random
import pandas as pd
import argparse
random.seed(48)

def extract_ans(response_str, options):
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
        r"The correct answer (?:for .*? )?is:?\s*([A-E])",
        r"^Option ([A-E]) is the correct answer"
        
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
    return ans_list
def main(args):
    random.seed(48)
    result_dir = 'results/medqa/MCQ'
    # names = ['meditron-7B']
    for name in args.models_been_eval:
        full_name = os.path.join(result_dir, name+'_MCQ_results.json')
        out_full_name = os.path.join(result_dir, name+'_MCQ_results_processed.json')
        outf = open(out_full_name,'w')
        model_name = name
        fail, ttl, corr, hit_corr = 0,0,0,0
        with open(full_name, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                ques = entry[1]
                direction = 'forward'
                options = re.findall(r"([A-E]): (.*?)(?:\t|\Z)",ques)
                options = {v[1]:v[0] for v in options}
                pos_ans = entry[2]
                pos_reply = entry[-1]
                pos_pred_list = extract_ans(pos_reply,options)
                if len(pos_pred_list) == 0:
                    # print(reply)
                    pos_pred = random.choice(['A','B','C','D','E'])
                    
                else:
                    pos_pred = pos_pred_list[0]
                entry += [pos_pred]
                outf.write(json.dumps(entry)+'\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_been_eval", type=list, default=['chatglm-6B','vicuna_7B','llama2-7B','bloomz-7b1-mt','meditron-7B','pulse-7B','vicuna_13B','llama2-70B', 'meditron-70B', 'clinicalcamel-70B', 'med42-70B','gemini-pro','gpt-3.5-turbo'])    
    args = parser.parse_args()
    main(args)