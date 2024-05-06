'''
    analysis the performance of LLMs at the axis of knowledge points.
    ONLY forward questions are used in this analysis
'''
import json
import argparse
import re
import random
import pandas as pd
import matplotlib.pyplot as plt

random.seed(48)

def analyze_results(models, iscot=False):
    if iscot:
        file_path = './results/medqa/{}/{}_{}_results_processed_cot.json'
    else:
        file_path = './results/medqa/{}/{}_{}_results_processed.json'
    results = {'Model':[], 'Comparison':[],'Discrimination':[],'Verification':[],'Rectification':[]}
    corr_learnt_final = {}
    for model in models:
        results['Model'] += [model]
        mcq, maq, tfq, rq_t, rq_f = 0,0,0,0,0
        tmp_result = {}
        for typ in ['MCQ','MAQ','TFQ','RQ']:
            data = []
            tmp_result[typ] = []
            with open(file_path.format(typ, model, typ),'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            assert len(data) == 795
            for i,item in enumerate(data):
                # key = item[0]
                # if key not in results:
                #     results[key] = []
                if typ == 'MCQ':
                    label = item[2]
                    answer = item[4]
                    corrs = [int(label==answer)]
                    mcq += int(label==answer)
                    tmp_result[typ].append(int(label==answer))
                    
                elif typ == 'MAQ':
                    label = item[3]
                    label_2 = item[4]
                    answer = item[-2]
                    answer_2 = item[-1]
                    label = set(label)
                    answer=  set(answer)
                    label_2 = set(label_2)
                    answer_2 = set(answer_2)
                    corr = int(label <= answer & answer <= label)
                    corr_2 = int(label_2 <= answer_2 & answer_2 <= label_2)
                    corrs = [corr, corr_2]
                    maq += corr + corr_2
                    tmp_result[typ].append((corr+corr_2)/2)
                elif typ == 'TFQ':
                    labels = []
                    answers = []
                    for i in range(5,9):
                        labels.append(item[i])
                    for i in range(13,17):    
                        answers.append(item[i])
                    corrs = [int(label==answer) for label, answer in zip(labels, answers)]
                    tfq += sum(corrs)
                    tmp_result[typ].append(sum(corrs)/4)
                elif typ == 'RQ':
                    labels = []
                    answers = []
                    for i in range(3,5):
                        labels.append(item[i])
                    for i in range(7,9):    
                        answers.append(item[i])
                    corrs = [int((label[0]==answer[0]) and (label[1] == answer[1])) for label, answer in zip(labels, answers)]
                    rq_t += corrs[0]
                    rq_f += corrs[1]
                    tmp_result[typ].append((corrs[0]+corrs[1]*4)/5)
        corr_learnt = [0,0,0]
        ttl = 0
        for c1,c2,c3,c4 in zip(tmp_result['MCQ'],tmp_result['MAQ'],tmp_result['TFQ'],tmp_result['RQ']):
            if c1 == 1:
                ttl += 1
                corr_learnt[0] += c2
                corr_learnt[1] += c3
                corr_learnt[2] += c4
        corr_learnt[0] /= ttl
        corr_learnt[1] /= ttl
        corr_learnt[2] /= ttl
        corr_learnt_final[model] = corr_learnt
                
        results['Comparison'] += [mcq / len(data)]
        results['Discrimination'] += [maq / len(data) / 2]
        results['Verification'] += [tfq / len(data) / 4]
        results['Rectification'] += [(rq_t+rq_f*4) / len(data)/ 5]
    return results,corr_learnt_final

def cal_relative_performance(results):
    random_guess = [1/5,1/31,1/2,1/5]
    labels = ['Comparison','Discrimination','Verification','Rectification']
    new_res = {}
    new_res['Model'] = results['Model']
    for i, l in enumerate(labels):
        rg = random_guess[i]
        new_res[l] = [(one-rg)/(1-rg) for one in results[l]]
    return new_res

def main(args):
    # keys = ['symptom','anatomic','procedure','medicine']
    # all_data = json.load(open('processed_data_icd4_1000.json','r'))
    # random_guess = [1/5,1/5,1/31,1/2]
    results, _ = analyze_results(args.models_been_eval, args.cot)
    df = pd.DataFrame(results)
    if args.cot:
        df.to_excel('results_cot.xlsx')
    else:
        df.to_excel('results_ao.xlsx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_been_eval", type=list, default=['chatglm-6B','vicuna_7B','llama2-7B','bloomz-7b1-mt','meditron-7B','pulse-7B','vicuna_13B','llama2-70B', 'meditron-70B', 'clinicalcamel-70B', 'med42-70B','gemini-pro','gpt-3.5-turbo'])    
    parser.add_argument("--cot", type=bool, default=True)
    args = parser.parse_args()
    main(args)