import os
import json
from tqdm import tqdm
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
import argparse
def chating(tmp_input, dev_data, neg, num_examples, model,tokenizer, config,task, repeat=0):
    new_len = -1
    while new_len < 0:
        demonstration = prepare_examples(dev_data, neg, num_examples,task)
        input_text = 'Please complete the final sample with the same format as the given examples.'+demonstration + tmp_input
        new_len = 2048-tokenizer.encode(input_text, return_tensors="pt").size(1)
        print(tokenizer.encode(input_text, return_tensors="pt").size(1))
        if new_len > 50:
            new_len = 50
        num_examples -= 1
    while True:
        try:
            inputs = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
            
            outputs = model.generate(inputs,max_new_tokens=new_len,eos_token_id=config.eos_token_id,repetition_penalty=1.1)
            break
        except:
            new_len -= 1
            if new_len <= 0:
                num_examples -= 1
                demonstration = prepare_examples(dev_data, neg, num_examples,task)
                input_text = demonstration + tmp_input
                new_len = 2048-len(tokenizer.tokenize(input_text))
            continue
                
        
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = reply[len(input_text):].strip()
    print(input_text)
    print('model\'s answer: '+reply)

    if 'sorry' in reply:
        if repeat == 5:
            return reply
        else:
            reply = chating(tmp_input, dev_data, neg, num_examples,model, tokenizer, config,task,repeat+1)
    return reply

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
            
            ans = 'true' if ans == 'T' else 'false'
            tmp += ques + '\nAnswer: '+ans+'\n\n'
    elif task == 'MCQ':
        for one in dev_data[:nums]:
            ques = one[1]
            ans = one[2]
            tmp += ques + '\nAnswer: '+ans+'\n\n'
    elif task == 'MAQ':
        for one in dev_data[:nums]:
            ques = one[1] if not neg else one[2]
            ans = one[3] if not neg else one[4]
            if isinstance(ans, list):
                ans = ','.join(ans)
            tmp += ques + '\nAnswer: '+ans+'\n\n'
    elif task == 'FIB':
        for one in dev_data[:nums]:
            ques = one[1]
            ans = one[2]
            if isinstance(ans, list):
                ans = ','.join(ans)
            tmp += ques + '\nAnswer: '+ans+'\n\n'
    elif task == 'RQ':
        for one in dev_data[:nums]:
            ques = one[1]
            ans = one[2]
            if ans[0] == 'T':
                tmp += ques + '\nAnswer: correct'+'\n\n'
            else:
                tmp += ques + '\nAnswer: incorrect, the correct answer is ' + ans[1]+'\n\n'

    return tmp

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = model.half()
    model = model.eval()
    
    cnt = -1
    for typ in ['MCQ','MAQ','TFQ','RQ']:
        dev_path = os.path.join('medqa/{}'.format(typ),'dev.json')
        path = os.path.join('medqa/{}'.format(typ), 'test.json')
        dev_data = load_data(dev_path)

        data = load_data(path)
        out_data = []
        
        if not os.path.exists('results/medqa/{}'.format(typ)):
            os.makedirs('results/medqa/{}'.format(typ))
        f = open('results/medqa/{1}/{0}_{1}_results.json'.format(args.model_name,typ),'a')

        for item in tqdm(data):
            cnt += 1
            if cnt < args.start:
                continue
            if typ == 'TFQ' or typ == 'TFQ_options':
                idx = item[0]
                ques = item[1]
                neg_ques = item[2]
                ques_2 = item[3]
                neg_ques_2 = item[4]
                input_text = ques + '\nAnswer: '
                neg_input_text = neg_ques + '\nAnswer: '
                input_text_2 = ques_2 + '\nAnswer: '
                neg_input_text_2 = neg_ques_2 + '\nAnswer: '
                reply = chating(input_text, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                neg_reply = chating(neg_input_text, dev_data, True, args.ntrain, model, tokenizer, config, typ)
                reply_2 = chating(input_text_2, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                neg_reply_2 = chating(neg_input_text_2, dev_data, True, args.ntrain, model, tokenizer, config, typ)
                out_data = item + [reply, neg_reply, reply_2, neg_reply_2]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
            elif typ == 'MAQ':
                idx = item[0]
                ques = item[1]
                neg_ques = item[2]
                input_text = ques + '\nAnswer: '
                neg_input_text = neg_ques + '\nAnswer: '
                reply = chating(input_text, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                neg_reply = chating(neg_input_text, dev_data, True, args.ntrain, model, tokenizer, config, typ)
                out_data = item + [reply, neg_reply]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
            elif typ == 'MCQ':
                idx = item[0]
                ques = item[1]
                input_text = ques + '\nAnswer: '
                reply = chating(input_text, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                out_data = item + [reply]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
            elif typ == 'RQ':
                idx = item[0]
                ques = item[1]
                ques_2 = item[2]
                input_text = ques + '\nAnswer: '
                input_text_2 = ques_2 + '\nAnswer: '
                reply = chating(input_text, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                reply_2 = chating(input_text_2, dev_data, False, args.ntrain, model, tokenizer, config, typ)
                out_data = item + [reply, reply_2]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
        f.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--model", type=str, default='path_of_model')
    parser.add_argument("--model_name", type=str, default='model_name')
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    main(args)

