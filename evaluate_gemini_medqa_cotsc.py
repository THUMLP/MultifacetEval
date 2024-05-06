import openai
import time
import os
import json
from tqdm import tqdm
import argparse
import google.generativeai as genai

def chating(tmp_input, dev_data, neg, num_examples,model, task,repeat=0):
    # try:
    max_token = 50
    while True:
        try:
            demonstration = prepare_examples(dev_data, neg, num_examples, task)
            input_text = 'Please complete the final example with the same format as the given examples.'+demonstration + tmp_input
            reply = model.generate_content(input_text)
            if str(reply.prompt_feedback.block_reason) != 'BlockReason.BLOCK_REASON_UNSPECIFIED':
                return str(reply.prompt_feedback.block_reason)
            reply = reply.text
            break
        except Exception as e:
            if repeat == 5:
                return 'time limit exceed'
            
            num_examples -= 1
            repeat += 1
            
    if 'sorry' in reply:
        if repeat == 5:
            return reply
        else:
            reply = chating(tmp_input, dev_data, neg, num_examples,model, task, repeat+1)
    return reply

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


def main(args):
    model = args.model
    GOOGLE_API_KEY='Your API KEY'
    model = genai.GenerativeModel('gemini-pro')
    genai.configure(api_key=GOOGLE_API_KEY)
    cnt = 0
    for typ in ['MCQ','MAQ','TFQ','RQ']:
        dev_path = os.path.join('medqa/{}'.format(typ),'dev_cot.json')
        path = os.path.join('medqa/{}'.format(typ), 'test.json')
        dev_data = load_data(dev_path)

        data = load_data(path)
        out_data = []
        
        if not os.path.exists('results/medqa/{}'.format(typ)):
            os.makedirs('results/medqa/{}'.format(typ))
        f = open('results/medqa/{1}/{0}_{1}_results_cot.json'.format(args.model,typ),'a')


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
                replys, neg_replys, replys_2, neg_replys_2 = [],[],[],[]
                for i in range(args.nchain):
                    reply = chating(input_text, dev_data, False, args.ntrain, model, typ)
                    neg_reply = chating(neg_input_text, dev_data, True, args.ntrain, model, typ)
                    reply_2 = chating(input_text_2, dev_data, False, args.ntrain, model, typ)
                    neg_reply_2 = chating(neg_input_text_2, dev_data, True, args.ntrain, model, typ)
                    replys.append(reply)
                    neg_replys.append(neg_reply)
                    replys_2.append(reply_2)
                    neg_replys_2.append(neg_reply_2)
                out_data = item + [replys, neg_replys, replys_2, neg_replys_2]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
            elif typ == 'MAQ':
                idx = item[0]
                ques = item[1]
                neg_ques = item[2]
                input_text = ques + '\nAnswer: '
                neg_input_text = neg_ques + '\nAnswer: '
                replys, neg_replys = [],[]
                for i in range(args.nchain):
                    reply = chating(input_text, dev_data, False, args.ntrain, model, typ)
                    neg_reply = chating(neg_input_text, dev_data, True, args.ntrain, model, typ)
                    replys.append(reply)
                    neg_replys.append(neg_reply)
                out_data = item + [replys, neg_replys]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
            elif typ == 'MCQ':
                idx = item[0]
                ques = item[1]
                input_text = ques + '\nAnswer: '
                replys = []
                for i in range(args.nchain):
                    reply = chating(input_text, dev_data, False, args.ntrain, model, typ)
                    replys.append(reply)
                out_data = item + [replys]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
            elif typ == 'RQ':
                idx = item[0]
                ques = item[1]
                ques_2 = item[2]
                input_text = ques + '\nAnswer: '
                input_text_2 = ques_2 + '\nAnswer: '
                replys, replys_2 = [],[]
                for i in range(args.nchain):
                    reply = chating(input_text, dev_data, False, args.ntrain, model, typ)
                    reply_2 = chating(input_text_2, dev_data, False, args.ntrain, model, typ)
                    replys.append(reply)
                    replys_2.append(reply_2)
                out_data = item + [replys, replys_2]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
        f.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--nchain", "-c", type=int, default=5)
    parser.add_argument("--model", type=str, default='gemini-pro')
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    main(args)
