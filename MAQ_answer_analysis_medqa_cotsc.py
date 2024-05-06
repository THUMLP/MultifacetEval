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
    response_str = response_str.replace('*','')
    
    pattern=[
        r"Therefore, the answer (?:are|is|would be|should be)\s*(?: a)?:?\s*\**\"?(?:option)?\s*([A-E](?:、|,|，|\s|[A-E])*)",
        r"(?:Therefore, )?(?:The|the) .*?(?:are|is|would be|should be)\s*(?: a)?:?\s*\**\"?(?:option)?\s*([A-E](?:、|,|，|\s|[A-E])*)",
        r"(?:The|the)[^\.]*(?:most|best) [^\.]* (?:is|would be|are|should be):?\s*([A-E](?:、|,|，|\s|[A-E])*)",
        r"answer is therefore: ([A-E](?:、|,|，|\s|[A-E])*)",
        r"^Options?:? ([A-E](?:、|,|，|\s|[A-E])*) (?:are|is) the correct answers?",
        r"The answer is that options? ([A-E](?:、|,|，|\s|[A-E])*)",
        r"(?:Option )?([A-E](?:、|,|，|\s|[A-E])*)[^\.]*(?:most|best)[^\.]*\.",
        r"^([A-E](?:、|,|，|\s|[A-E])*)",
        r"^Options?:?\s*([A-E](?:、|,|，|\s|[A-E])*)",
        r"Answers?:?\s*([A-E](?:、|,|，|\s|[A-E])*)",
        r"^[^\.]*([A-E](?:、|,|，|\s|[A-E])*)[^\.]*\.",
    ]
    all_pattern = r"(?:Therefore, )?(?:all|All) of the (?:listed )?options? .*?(?:are|is|would be|can)"
    # for ans_idx, ans_text in options.items():
    #     text = '{}: {}'.format(ans_idx, ans_text)
    #     text_2 = '{}. {}'.format(ans_idx, ans_text)
    #     response_str = response_str.replace(text, ans_idx)
    #     response_str = response_str.replace(text_2, ans_idx)
    inv_options = {v:k for k,v in options.items()}
    for key in inv_options:
        if inv_options[key]+f': {key}' in response_str:
            response_str = response_str.replace(inv_options[key]+f': {key}', inv_options[key])
        if key in response_str:
            response_str = response_str.replace(key, inv_options[key])
        if key.lower() in response_str:
            response_str = response_str.replace(key.lower(), inv_options[key])
        if key[0].lower()+key[1:] in response_str:
            response_str = response_str.replace(key[0].lower()+key[1:], inv_options[key])
        
    ans_list=[]
    response_str = response_str.strip()
    pattern_idx = -1
    if len(response_str) == 0:
        return []
    # if response_str[0] in ['A','B','C','D']:
    #     ans_list.append(response_str[0])
    for i,p in enumerate(pattern):
        if len(ans_list)==0:
            ans_list=re.findall(p,response_str,re.DOTALL)
            pattern_idx = i
        else:
            break
    if len(ans_list) > 0:
        ans_list = ans_list[0]
        for spt in ['，','、',',','\n']:
            if spt in ans_list:
                ans_list = ans_list.split(spt)
                ans_list = [one.strip() for one in ans_list]
                break
    else:
        res = re.search(all_pattern, response_str, re.DOTALL)
        if res:
            ans_list = ['A','B','C','D','E']
    new_ans_list = [one for one in ans_list if one in ['A','B','C','D','E']]
    # if len(new_ans_list) == 0:
    #     # if not response_str.startswith('Question'):
    #     if len(ans_list) > 0:
    #         print('y')
    #     print(response_str)
    # if len(ans_list) > 0:
    #     print(response_str)
    return new_ans_list
def option_vote(preds):
    ans_count = [0,0,0,0,0]
    num_valid = 0
    for item in preds:
        if len(item) > 0:
            num_valid += 1
        for ans in item:
            ans_count[ord(ans)-ord('A')] += 1
    if sum(ans_count) == 0:
        return []
    else:
        final_ans = []
        for i in range(5):
            if ans_count[i] >= num_valid/2:
                final_ans.append(chr(i+ord('A')))
        return final_ans
    
def answer_vote(preds):
    ans_count = {}
    for item in preds:
        item = sorted(item)
        item_text = ','.join(item)
        if item_text == '':
            continue
        if item_text not in ans_count:
            ans_count[item_text] = 0
        ans_count[item_text] += 1
    if len(ans_count) == 0:
        return []
    else:
        max_count = -1
        ans_candidate = []
        for key in ans_count:
            if ans_count[key] > max_count:
                max_count = ans_count[key]
        for key in ans_count:
            if ans_count[key] == max_count:
                ans_candidate.append(key)
        final_ans = random.choice(ans_candidate).split(',')
        return final_ans
def main(args):
    random.seed(48)
    result_dir = 'results/medqa/MAQ'
    # names = ['chatglm-6B','meditron-7B','pulse-7B','vicuna_7B','llama2-7B','bloomz-7b1-mt','vicuna_13B','med42-70B','llama2-70B','meditron-70B','clinicalcamel-70B','gemini-pro','gpt-3.5-turbo']
    mode = 1 # 0: option vote, 1: answer vote
    for name in args.models_been_eval:
        full_name = os.path.join(result_dir, name+'_MAQ_results_cot.json')
        out_full_name = os.path.join(result_dir, name+'_MAQ_results_processed_cot.json')
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
                neg_replys = entry[-1]
                pos_replys = entry[-2]
                pos_preds, neg_preds = [],[]
                for neg_reply, pos_reply in zip(neg_replys, pos_replys):
                    tmp_pos_pred_list = extract_ans(pos_reply, pos_options)
                    tmp_neg_pred_list = extract_ans(neg_reply, neg_options)
                    pos_preds.append(tmp_pos_pred_list)
                    neg_preds.append(tmp_neg_pred_list)
                if mode == 0:
                    pos_pred_list = option_vote(pos_preds)
                    neg_pred_list = option_vote(neg_preds)
                else:
                    pos_pred_list = answer_vote(pos_preds)
                    neg_pred_list = answer_vote(neg_preds)
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

