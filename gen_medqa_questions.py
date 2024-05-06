'''
    generating MAQ,TFQ,RQ questions from the original 800 matched USMLE questions
'''
from medcat.cat import CAT
import json
import random
import copy
import os
from itertools import product
import numpy as np
random.seed(48)
cat = CAT.load_model_pack('Your MedCAT PATH')
synonyms = json.load(open('concept_attributes.json','r'))

def get_synonym(text, entities):
    '''
        get synonym of the given sentence
    '''
    all_list = []
    original = []
    for key in entities['entities']:
        item = entities['entities'][key]
        try:
            cui = item['cui']
            syms = synonyms[cui]
            new_syms = [item['source_value']]
            tmp = [item['source_value'].lower()]
            assert item['source_value'] in text
            for one in syms:
                if one.lower() not in tmp:
                    new_syms += [one]
                    tmp += [one.lower()]
            syms = new_syms
        except:
            continue
        detected_name = item['source_value'].lower()
        while '~' in detected_name:
            detected_name = detected_name.replace('~',' ')
        new_other_syms = []
        for one in syms:
            if ',' not in one and '+' not in one and '>' not in one:
                new_other_syms.append(one)
        other_syms = new_other_syms
        all_list.append(other_syms)
        # sym = random.choice(other_syms).lower()
        original.append(item['source_value'])
        # assert item['']
        # text = text.lower().replace(detected_name, sym).capitalize()
    if len(original) == 0:
        return [text]
    all_combs = list(product(*all_list))
    all_texts = []
    sorted_ids = np.argsort([-len(one) for one in original])
    
    for comb in all_combs:
        new_text = copy.deepcopy(text)
        for i in sorted_ids:
            # if original[i] not in new_text:
            #     print('{}\t{}'.format(new_text, original[i]))
            new_text = new_text.replace(original[i], comb[i])
        new_text = new_text.capitalize()
        all_texts.append(new_text)
    return all_texts

import copy
from tqdm import tqdm
path = 'rewrite.jsonl'
# valid_keys = ['Disease or Syndrome','']
out_data = {}
with open(path,'r') as f:
    for line in tqdm(f):
        entry = json.loads(line.strip())
        options = entry['options']
        # Test it
        answer = entry['answer']
        answer_text = options[answer]
        answer_before = copy.deepcopy(answer_text)
        entities = cat.get_entities(answer_text)
        
        answer_texts = get_synonym(answer_before, entities)
        out_data[answer_text] = answer_texts
json.dump(out_data, open('synonyms_of_options.json','w'),indent=4)
            

# Download the model_pack from the models section in the github repo.
# cat = CAT.load_model_pack('/home/zhouyx/medcat/umls_self_train_model_pt2ch_3760d588371755d0.zip')
# concept_types = json.load(open('concept_semantic_types.json','r'))
# synonym = json.load(open('concept_attributes.json','r'))
path = 'rewrite.jsonl'
mcq_data, mcq_dev_data, maq_data, maq_dev_data, tfq_data,\
      tfq_dev_data, fib_data, fib_dev_data, tfq_2_data, tfq_2_dev_data, ar_dev_data, ar_data = [], [],[],[],[],[],[],[],[],[],[],[]
all_data = []
with open(path,'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        all_data.append(entry)
random.shuffle(all_data)
synonyms_options = json.load(open('synonyms_of_options.json','r'))
for i,entry in enumerate(all_data):
    options = entry['options']
    # Test it
    answer = entry['answer']
    answer_text = options[answer]
    same_meaning_options = synonyms_options[answer_text]
    # original_ques:
    mcq_ques = 'Question: ' + entry['ques']
    options_text = ['{}: {}'.format(key, options[key]) for key in options]
    mcq_ques += '\n' + 'Options: '+'\t'.join(options_text)
    # MAQ
    corr_num = random.randint(1, 4)
    other_ans = list(set(['A','B','C','D','E'])-{answer})
    corr_ans = [answer]
    new_options = copy.deepcopy(options)
    if corr_num > 1:
        if len(same_meaning_options) >= corr_num-1:
            same_meaning_options = random.sample(same_meaning_options,k=corr_num-1)
        else:
            same_meaning_options = same_meaning_options + random.choices(same_meaning_options, k=corr_num-1-len(same_meaning_options))

        other_corr = random.sample(other_ans, k=corr_num-1)
        corr_ans += other_corr
        for j,one in enumerate(other_corr):
            new_options[one] = same_meaning_options[j]
    
    maq_ques = 'Question: '+entry['maq_ques']
    neg_maq_ques = 'Question: '+entry['neg_maq_ques']
    new_options_text = ['{}: {}'.format(key, new_options[key]) for key in new_options]
    maq_ques += '\n' + 'Options: '+ '\n'.join(new_options_text)
    neg_maq_ques += '\n' + 'Options: '+ '\n'.join(new_options_text)
    maq_ans = corr_ans
    neg_maq_ans = list(set(['A','B','C','D','E'])-set(corr_ans))
    corr_ans.sort()
    neg_maq_ans.sort()
    other = random.choice(other_ans)
    other_text = options[other]
    ar_ques_p = mcq_ques + '. Alice\'s answer: {}. Please determine whether her answer is correct, and if it\'s incorrect, provide the correct answer.'.format(answer)
    ar_ques_n = mcq_ques + '. Alice\'s answer: {}. Please determine whether her answer is correct, and if it\'s incorrect, provide the correct answer.'.format(other)
    # tfq
    tfq_p = 'Question: '+entry['tfq_ques'].replace('[option_text]', answer_text)
    neg_tfq_n = 'Question: '+entry['neg_tfq_ques'].replace('[option_text]', answer_text)
    tfq_n = 'Question: '+entry['tfq_ques'].replace('[option_text]', other_text)
    neg_tfq_p = 'Question: '+entry['neg_tfq_ques'].replace('[option_text]', other_text)
    if i <5:
        one = random.choice([[i,tfq_p,neg_tfq_n,'T','F'],[i,tfq_n,neg_tfq_p,'F','T']])
        one_2 = random.choice([[i, ar_ques_p, ['T',answer]],[i, ar_ques_n,['F',answer]]])
        tfq_dev_data.append(one)
        mcq_dev_data.append([i, mcq_ques, answer])
        maq_dev_data.append([i, maq_ques, neg_maq_ques, list(corr_ans), list(neg_maq_ans)])
        ar_dev_data.append(one_2)
    else:
        tfq_data.append([i, tfq_p, neg_tfq_n, tfq_n, neg_tfq_p,  'T','F','F','T'])
        mcq_data.append([i, mcq_ques, answer])
        maq_data.append([i, maq_ques, neg_maq_ques, list(corr_ans), list(neg_maq_ans)])
        ar_data.append([i, ar_ques_p, ar_ques_n, ['T',answer], ['F',answer]])
for name in ['TFQ','MCQ','MAQ','RQ']:
    if not os.path.exists('medqa/{}'.format(name)):
        os.makedirs('medqa/{}'.format(name))
json.dump(tfq_dev_data,open('medqa/TFQ/dev.json','w'),indent=4)
json.dump(tfq_data,open('medqa/TFQ/test.json','w'),indent=4)

json.dump(mcq_dev_data,open('medqa/MCQ/dev.json','w'),indent=4)
json.dump(mcq_data,open('medqa/MCQ/test.json','w'),indent=4)

json.dump(maq_dev_data,open('medqa/MAQ/dev.json','w'),indent=4)
json.dump(maq_data,open('medqa/MAQ/test.json','w'),indent=4)
        
json.dump(ar_dev_data,open('medqa/RQ/dev.json','w'),indent=4)
json.dump(ar_data,open('medqa/RQ/test.json','w'),indent=4)