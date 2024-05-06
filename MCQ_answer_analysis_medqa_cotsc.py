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
    # if 'The most likely diagnosis is contact dermatitis.' in response_str:
    #     print('y')
    for key in options:
        if key in response_str:
            response_str = response_str.replace(key, options[key])
        if key.lower() in response_str:
            response_str = response_str.replace(key.lower(), options[key])
        if key[0].lower()+key[1:] in response_str:
            response_str = response_str.replace(key[0].lower()+key[1:], options[key])
    pattern=[
        r"Therefore, the answer (?:are|is|would be|should be)\s*(?: a)?:?\s*\**\"?(?:option)?\s*([A-E])",
        r"(?:Therefore, )?(?:The|the)[^\.]*(?:are|is|would be|should be)\s*(?: a)?:?\s*\**\"?(?:option)?\s*([A-E])",
        r"(?:The|the)[^\.]*(?:most|best) [^\.]* (?:is|would be|are|should be)\s*:?(?:an|the)?\s*([A-E])",
        # r"answer is therefore: ([A-E])",
        # r"^Options?:? ([A-E]) (?:are|is) the correct answers?",
        r"The answer is that options? ([A-E])",
        r"(?:Option )?([A-E])[^\.]*(?:most|best)[^\.]*\.",
        r"(?:are|is) consistent with ([A-E])",
        r"^([A-E])",
        r"^[^\.]*([A-E])\.",
        # r"[^\.]*(?:include|involve)?s?\s*:?(?:an|the)?([A-E])(?:,|\.)",
        # r"^Options?:?\s*([A-E])",
        # r"Answers?:?\s*([A-E])",
    ]
    ans_list=[]
    response_str = response_str.strip()
    if len(response_str) == 0:
        return '', -1
    # if response_str[0] in ['A','B','C','D','E']:
    #     ans_list.append(response_str[0])
    pattern_idx = -1
    for i,p in enumerate(pattern):
        if len(ans_list)==0:
            ans_list=re.findall(p,response_str,re.DOTALL)
            if len(ans_list) > 0:
                pattern_idx = i
        else:
            break
    
    if len(ans_list) == 0:
        # print(response_str)
        ans_list = ''
    else:
        # if pattern_idx == 9:
        ans_list = ans_list[0]
    # if pattern_idx == 2:
    #     print(response_str)
    return ans_list, pattern_idx
def main(args):
    random.seed(48)
    result_dir = 'results/medqa/MCQ'
    # names = ['chatglm-6B','meditron-7B','pulse-7B','vicuna_7B','llama2-7B','bloomz-7b1-mt','vicuna_13B','med42-70B','llama2-70B','meditron-70B','clinicalcamel-70B','gemini-pro','gpt-3.5-turbo']
    # names = ['chatglm-6B','meditron-7B','pulse-7B','vicuna_7B','llama2-7B','bloomz-7b1-mt','llama2-70B','meditron-70B','clinicalcamel-70B','gpt-3.5-turbo','gemini-pro']

    pattern_cnt = {}
    pattern_corr = {}
    answers = []
    for name in args.models_been_eval:
        full_name = os.path.join(result_dir, name+'_MCQ_results_cot.json')
        out_full_name = os.path.join(result_dir, name+'_MCQ_results_processed_cot.json')
        outf = open(out_full_name,'w')
        model_name = name
        fail, ttl, corr, hit_corr = 0,0,0,0
        with open(full_name, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                ques = entry[1]
                direction = 'forward'
                options = re.findall(r"([A-E]): (.*?)(?:\t|\n|\Z)",ques)
                options = {v[1]:v[0] for v in options}
                pos_ans = entry[2]
                pos_replys = entry[-1]
                pos_pred_list = []
                for pos_reply in pos_replys:
                    ans, res = extract_ans(pos_reply,options)
                    
                    pos_pred_list.append(ans)
                    if res == -1:
                        continue
                    if res != -1 and res not in pattern_cnt:
                        pattern_cnt[res] = 0
                        pattern_corr[res] = 0
                    pattern_cnt[res] += 1
                    if res != -1:
                        if ans == pos_ans:
                            pattern_corr[res] += 1
                        # elif res == 2:
                        #     print(pos_reply)
                        
                        
                answer_counts = [0,0,0,0,0]
                for tmp_pos_pred in pos_pred_list:
                    # if len(tmp_pos_pred) == 0:
                    #     results[direction]['pos'][0] += 1
                        # print(reply)
                    if not len(tmp_pos_pred) == 0:
                        answer_counts[ord(tmp_pos_pred)-ord('A')] += 1
                
                if sum(answer_counts) == 0: # mismatch
                    pos_pred = random.choice(['A','B','C','D','E'])
                else:
                    pos_pred_candidates = [i for i in range(5) if answer_counts[i] == max(answer_counts)]
                    pos_pred = random.choice(pos_pred_candidates)
                    pos_pred = chr(pos_pred+ord('A'))
                answers.append([pos_ans, pos_pred,int(pos_pred==pos_ans)])
                entry += [pos_pred]
                outf.write(json.dumps(entry)+'\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_been_eval", type=list, default=['chatglm-6B','vicuna_7B','llama2-7B','bloomz-7b1-mt','meditron-7B','pulse-7B','vicuna_13B','llama2-70B', 'meditron-70B', 'clinicalcamel-70B', 'med42-70B','gemini-pro','gpt-3.5-turbo'])    
    args = parser.parse_args()
    main(args)
# for key in pattern_cnt:
#     print(f'{key}\t{pattern_corr[key]/pattern_cnt[key]}')