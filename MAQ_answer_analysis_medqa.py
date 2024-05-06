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
    if ' and ' in response_str:
        response_str = response_str.replace(' and ',',')
    pattern=[
        r"^([A-E](?:、|,|，|\s|[A-E])*)",
        r"^Options?:? ([A-E](?:、|,|，|\s|[A-E])*)",
        r"Answers?:?\s*([A-E](?:、|,|，|\s|[A-E])*)",
        r"(?:Therefore, )?(?:The|the) (?:correct |best )?answers? (?:for .*? )?(?:are|is):?\s*\**\"?(?:option)?\s*([A-E](?:、|,|，|\s|[A-E])*)",
        r"^Options?:? ([A-E](?:、|,|，|\s|[A-E])*) (?:are|is) the correct answers?"
        r"The answer is that options? ([A-E](?:、|,|，|\s|[A-E])*)",

    ]
    # for ans_idx, ans_text in options.items():
    #     text = '{}: {}'.format(ans_idx, ans_text)
    #     text_2 = '{}. {}'.format(ans_idx, ans_text)
    #     response_str = response_str.replace(text, ans_idx)
    #     response_str = response_str.replace(text_2, ans_idx)
    inv_options = {v:k for k,v in options.items()}
    for key in inv_options:
        if key in response_str:
            response_str = response_str.replace(key, inv_options[key])
    ans_list=[]
    response_str = response_str.strip()
    if len(response_str) == 0:
        return []
    # if response_str[0] in ['A','B','C','D']:
    #     ans_list.append(response_str[0])
    for p in pattern:
        if len(ans_list)==0:
            ans_list=re.findall(p,response_str,re.DOTALL)
        else:
            break
    if len(ans_list) > 0:
        ans_list = ans_list[0]
        for spt in ['，','、',',','\n']:
            if spt in ans_list:
                ans_list = ans_list.split(spt)
                ans_list = [one.strip() for one in ans_list]
                break
    ans_list = [one for one in ans_list if one in ['A','B','C','D','E']]
    # if len(ans_list) == 0:
    #     # if not response_str.startswith('Question'):
    #     print(response_str)
    # if len(ans_list) > 0:
    #     print(response_str)
    return ans_list
def main(args):
    random.seed(48)
    result_dir = 'results/medqa/MAQ'

    for name in args.models_been_eval:
        full_name = os.path.join(result_dir, name+'_MAQ_results.json')
        out_full_name = os.path.join(result_dir, name+'_MAQ_results_processed.json')
        outf = open(out_full_name,'w')
        model_name = name
        fail, ttl, corr, hit_corr = 0,0,0,0
        with open(full_name, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())

                ques = entry[1]
                neg_ques = entry[2]
                direction = 'forward'
                pos_ans = entry[3]
                pos_options = re.findall(r"([A-E]): (.*?)(?:\n|\Z)",ques)
                pos_options = {v[0]:v[1] for v in pos_options}
                neg_ans = entry[4]
                neg_options = re.findall(r"([A-E]): (.*?)(?:\n|\Z)",neg_ques)
                neg_options = {v[0]:v[1] for v in neg_options}
                neg_reply = entry[-1]
                pos_reply = entry[-2]
                pos_pred_list = extract_ans(pos_reply, pos_options)
                neg_pred_list = extract_ans(neg_reply, neg_options)
                if len(pos_pred_list) == 0:
                    
                    pos_pred = []
                    
                else:
                    pos_pred = pos_pred_list
                    pos_pred = set(pos_pred)
                    pos_ans = set(pos_ans)
                if len(neg_pred_list) == 0:
                    neg_pred = []
                    
                else:
                    neg_pred = neg_pred_list
                    neg_pred = set(neg_pred)
                    neg_ans = set(neg_ans)
                entry += [list(pos_pred), list(neg_pred)]
                outf.write(json.dumps(entry,ensure_ascii=False))
                outf.write('\n')
                outf.flush()
        outf.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_been_eval", type=list, default=['chatglm-6B','vicuna_7B','llama2-7B','bloomz-7b1-mt','meditron-7B','pulse-7B','vicuna_13B','llama2-70B', 'meditron-70B', 'clinicalcamel-70B', 'med42-70B','gemini-pro','gpt-3.5-turbo'])    
    args = parser.parse_args()
    main(args)