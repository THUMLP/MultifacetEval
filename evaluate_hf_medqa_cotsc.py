import openai
import time
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import json
from tqdm import tqdm
import argparse
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    AutoModelForCausalLM,
    set_seed,
)

import argparse
'''
    "感染性黄疸": {
        "name": "感染性黄疸",
        "people": [],
        "typical_age": [],
        "special_age": [],
        "symptom": [
            "黄疸",
            "尿色异常",
            "腹胀",
            "腹痛",
            "乏力",
            "眼黄",
            "食欲不振",
            "纳差",
            "面黄",
            "发热",
            "精神萎靡",
            "瘙痒",
            "皮肤黄染"
        ],
        "accompany_symptom": [],
        "differential_symptom": [
            "皮肤",
            "巩膜"
        ],
        "physical_examination": "皮肤及黏膜检查（皮肤及巩膜黄染）",
        "anatomic": [
            "皮肤",
            "巩膜"
        ],
        "affected_parts": [
            "皮肤",
            "黏膜",
            "巩膜"
        ],
        "treatment_principle": [
            "病因治疗",
            "对症治疗",
            "抗感染"
        ],
        "secondary_disease": [],
        "procedure": [],
        "medicine": [
            "还原性谷胱甘肽",
            "腺苷蛋氨酸",
            "水飞蓟素",
            "多烯磷脂胆碱",
            "前列地儿"
        ],
        "abnormal_ancillary": "腹部CT检查（胆管扩张）",
        "abnormal_lab": "肝功能检查（血清总胆红素增高 乳酸脱氢酶升高 谷草转氨酶升高 碱性磷酸酶升高 ） 尿常规（尿胆原升高）",
        "department": [
            "肝胆胰脾外科",
            "感染科"
        ],
        "severity": 4,
        "zh_nickname": [],
        "en_abbr": "无",
        "body_system": [
            "消化系统"
        ]
    },
'''
def chating(tmp_inputs, dev_data, num_examples, model,tokenizer, config,task, repeat=0):
    new_len = -1
    while new_len < 0:
        input_texts = batch_prepare_inputs(tmp_inputs,dev_data=dev_data, nums=num_examples, task=task)
        input_lens = [len(one) for one in input_texts]
        max_len = -1
        max_ids = -1
        for i, one in enumerate(input_lens):
            if one > max_len:
                max_len = one
                max_ids = i

        new_len = 2048-tokenizer.encode(input_texts[i], return_tensors="pt").size(1)
        # print(tokenizer.encode(input_text, return_tensors="pt").size(1))
        if new_len > 512:
            new_len = 512
        num_examples -= 1
    # while True:
        # try:
    print('encode')
    
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to("cuda")
    print('gen')
    outputs = model.generate(**inputs,num_return_sequences=args.nchain,max_new_tokens=new_len,eos_token_id=config.eos_token_id,do_sample=True,top_p=0.8, temperature=0.8)
    # break
        # except:
        #     new_len -= 1
        #     if new_len <= 0:
        #         num_examples -= 1
        #         demonstration = prepare_examples(dev_data, neg, num_examples,task)
        #         input_text = demonstration + tmp_input
        #         new_len = 2048-len(tokenizer.tokenize(input_text))
        #         if new_len > 512:
        #             new_len = 512
        #     continue
                
    preds= tokenizer.batch_decode(outputs, skip_special_tokens=True)
    replys = []
    for i in range(len(input_texts)):
        tmp_replys = []
        for j in range(args.nchain):
            tmp_replys.append(preds[i*args.nchain+j][len(input_texts[i]):].strip())
        replys.append(tmp_replys)
    # print(input_text)
    # print('model\'s answer: '+reply)

    # if 'sorry' in reply:
    #     if repeat == 5:
    #         return reply
    #     else:
    #         reply = chating(tmp_input, dev_data, neg, num_examples,model, tokenizer, config,task,repeat+1)
    return replys

input_file_name = 'disease_description'

# prompt = json.load(open('new_prompts.json','r'))
exp_prompt = {
        #     'people':'{}的常见患病人群包括{}。',
        #   'typical_age':'{}的好发年龄包括{}。',
        #   'special_age':'{}的特发年龄包括{}。',
          'symptom':'{}的常见症状包括{}。',
        #   'accompany_symptom':'{}的伴随症状包括{}。',
        #   'differential_symptom':'{}的鉴别性症状包括{}。',
        #   'physical_examination':'{}的体格检查为{}。',
          'anatomic':'{}的解剖部位包括{}。',
        #   'affected_parts':'{}的影响部位包括{}。',
        #   'treatment_principle':'{}的治疗原则包括{}。',
        #   'secondary_disease':'{}的常见继发疾病包括{}。',
          'procedure':'{}的常见手术治疗名称包括{}。',
          'medicine':'{}的常见治疗药物名称包括{}。',
        #   'abnormal_ancillary':'{}的辅助检查异常结果为{}。',
        #   'abnormal_lab':'{}的实验室检查异常结果为{}。',
        #   'department':'{}涉及的科室包括{}。',
        #   'severity':'{}的危重等级是{}。',
        #   'zh_nickname':'{}的中文别名包括{}。',
        #   'en_abbr':'{}的英文缩写是{}。',
        #   'body_system':'{}涉及的相关人体系统包括{}。',
          }
ques_prompt = {
        #     'people':'{}的常见患病人群包括',
        #   'typical_age':'{}的好发年龄包括',
        #   'special_age':'{}的特发年龄包括',
          'symptom':'{}的常见症状包括',
        #   'accompany_symptom':'{}的伴随症状包括',
        #   'differential_symptom':'{}的鉴别性症状包括',
        #   'physical_examination':'{}的体格检查为',
          'anatomic':'{}的解剖部位包括',
        #   'affected_parts':'{}的影响部位包括',
        #   'treatment_principle':'{}的治疗原则包括',
        #   'secondary_disease':'{}的常见继发疾病包括',
          'procedure':'{}的常见手术治疗名称包括',
          'medicine':'{}的常见治疗药物名称包括',
        #   'abnormal_ancillary':'{}的辅助检查异常结果为',
        #   'abnormal_lab':'{}的实验室检查异常结果为',
        #   'department':'{}涉及的科室包括',
        #   'severity':'{}的危重等级是',
        #   'zh_nickname':'{}的中文别名包括',
        #   'en_abbr':'{}的英文缩写是',
        #   'body_system':'{}涉及的相关人体系统包括',
          }
# for i, line in enumerate(fin):
    
#     input_text = prompt + line.strip()
# model = 'gpt-3.5-turbo'

def load_data(path):
    return json.load(open(path,'r'))

def prepare_examples(dev_data,neg,nums,task):
    tmp = ''
    if task == 'TFQ' or task == 'TFQ_options':
        for one in dev_data[:nums]:
            ques = one[1] if not neg else one[2]
            ans = one[3] if not neg else one[4]
            cot = one[5] if not neg else one[6]
            if cot[-1] != '.':
                cot += '.'
            ans = 'true' if ans == 'T' else 'false'
            tmp += ques + f'\nAnswer: {cot} Therefore, the answer is: '+ans+'\n\n'
    elif task == 'MCQ':
        for one in dev_data[:nums]:
            ques = one[1]
            ans = one[2]
            cot = one[3]
            if cot[-1] != '.':
                cot += '.'
            tmp += ques + f'\nAnswer: {cot} Therefore, the answer is: '+ans+'\n\n'
    elif task == 'MAQ':
        for one in dev_data[:nums]:
            ques = one[1] if not neg else one[2]
            ans = one[3] if not neg else one[4]
            cot = one[5] if not neg else one[6]
            if cot[-1] != '.':
                cot += '.'
            if isinstance(ans, list):
                ans = ','.join(ans)
            tmp += ques + f'\nAnswer: {cot} Therefore, the answer is: '+ans+'\n\n'
    elif task == 'RQ':
        for one in dev_data[:nums]:
            ques = one[1]
            ans = one[2]
            cot = one[3]
            if cot[-1] != '.':
                cot += '.'
            if ans[0] == 'T':
                tmp += ques + f'\nAnswer: {cot} Therefore, the answer is: correct'+'\n\n'
            else:
                tmp += ques + f'\nAnswer: {cot} Therefore, the answer is: incorrect, the correct answer is ' + ans[1]+'\n\n'

    return tmp

def batch_prepare_inputs(inputs, dev_data, nums, task):
    pos_example = prepare_examples(dev_data,False,nums,task)
    neg_example = prepare_examples(dev_data, True, nums, task)
    examples = []
    for one in inputs:
        if one[1]:
            examples += ['Please complete the final sample with the same format as the given examples.'+ neg_example + one[0]]
        else:
            examples += ['Please complete the final sample with the same format as the given examples.'+ pos_example + one[0]]
    return examples    


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", 
    # attn_implementation="flash_attention_2",
      device_map="auto")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = model.bfloat16()
    # model = model.cuda()
    model = model.eval()
    
    cnt = -1
    # backward_path = os.path.join('data/{}'.format(typ), 'backward_questions.json')
    for typ in ['MCQ','MAQ','TFQ','RQ']:
        dev_path = os.path.join('medqa/{}'.format(typ),'dev_cot.json')
        path = os.path.join('medqa/{}'.format(typ), 'test.json')
        dev_data = load_data(dev_path)

        data = load_data(path)
        out_data = []
        
        if not os.path.exists('results/medqa/{}'.format(typ)):
            os.makedirs('results/medqa/{}'.format(typ))
        f = open('results/medqa/{1}/{0}_{1}_results_cot.json'.format(args.model_name,typ),'a')


        task_pool = []
        item_pool = []
        for item in tqdm(data):
            cnt += 1
            if cnt < args.start:
                continue
            if typ == 'TFQ':
                idx = item[0]
                ques = item[1]
                neg_ques = item[2]
                ques_2 = item[3]
                neg_ques_2 = item[4]
                input_text = ques + '\nAnswer: '
                neg_input_text = neg_ques + '\nAnswer: '
                input_text_2 = ques_2 + '\nAnswer: '
                neg_input_text_2 = neg_ques_2 + '\nAnswer: '
                task_pool += [[input_text, False],[neg_input_text, True],[input_text_2,False],[neg_input_text_2,True]]
                item_pool += [item]
                # replys = chating(input_text, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                # neg_replys = chating(neg_input_text, dev_data, True, args.ntrain, model, tokenizer, config, typ)
                # replys_2 = chating(input_text_2, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                # neg_replys_2 = chating(neg_input_text_2, dev_data, True, args.ntrain, model, tokenizer, config, typ)
                # out_data = item + [replys, neg_replys, replys_2, neg_replys_2]
                # f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                # f.flush()
            elif typ == 'MAQ':
                idx = item[0]
                ques = item[1]
                neg_ques = item[2]
                input_text = ques + '\nAnswer: '
                neg_input_text = neg_ques + '\nAnswer: '
                task_pool += [[input_text, False],[neg_input_text, True]]
                item_pool += [item]
                # replys = chating(input_text, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                # neg_replys = chating(neg_input_text, dev_data, True, args.ntrain, model, tokenizer, config, typ)

                # out_data = item + [replys, neg_replys]
                # f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                # f.flush()
            elif typ == 'MCQ':
                idx = item[0]
                ques = item[1]
                input_text = ques + '\nAnswer: '
                task_pool += [[input_text, False]]
                item_pool += [item]
                # replys = chating(input_text, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                # out_data = item + [replys]
                # f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                # f.flush()
            elif typ == 'RQ':
                idx = item[0]
                ques = item[1]
                ques_2 = item[2]
                input_text = ques + '\nAnswer: '
                input_text_2 = ques_2 + '\nAnswer: '
                task_pool += [[input_text, False],[input_text_2, False]]
                item_pool += [item]
                # replys = chating(input_text, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                # replys_2 = chating(input_text_2, dev_data, False, args.ntrain, model, tokenizer, config, typ)

                # out_data = item + [replys, replys_2]
                # f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                # f.flush()
        num_iters = len(task_pool)//args.nbatch+1
        all_results = []
        ncnt = 0
        if typ == 'TFQ':
            loop=4
        elif typ == 'MAQ':
            loop=2
        elif typ == 'MCQ':
            loop=1
        elif typ == 'RQ':
            loop=2
        for i in tqdm(range(num_iters)):
            batch = task_pool[i*args.nbatch:(i+1)*args.nbatch]
            results = chating(batch, dev_data, args.ntrain, model,tokenizer, config,typ)
            all_results += results
            while len(all_results) >= loop:
                item = item_pool[ncnt]
                ncnt += 1
                item += all_results[:loop]
                all_results = all_results[loop:]
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
                f.flush()
        f.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--nchain", "-c", type=int, default=5)
    parser.add_argument("--nbatch", "-b", type=int, default=1)
    parser.add_argument("--model", type=str, default='path_of_model')
    parser.add_argument("--model_name", type=str, default='model_name')
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    main(args)

