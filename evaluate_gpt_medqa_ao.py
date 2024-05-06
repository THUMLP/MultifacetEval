import openai
import time
import os
import json
from tqdm import tqdm
import argparse
openai.api_key = 'Your API KEY'
openai.api_base = "Your API BASE"

def chating(tmp_input, dev_data, neg, num_examples,model, task,repeat=0):
    # try:
    max_token = 200
    try:
        demonstration = prepare_examples(dev_data, neg, num_examples, task)
        input_text = 'Please complete the final sample with the same format as the given examples.'+demonstration + tmp_input
        reply = openai.ChatCompletion.create(model=model,
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text},      
        ],
        max_tokens=max_token
        # temperature=0
        )
        if reply.choices[0]['finish_reason'] == 'content_filter':
            return 'content filter'
        reply = reply.choices[0]['message']['content']
        
    except Exception:
        if repeat == 5:
            return 'time limit exceed'
        
        time.sleep(1)
        reply = chating(tmp_input, dev_data, neg, num_examples,model, task, repeat+1)
    
    if 'sorry' in reply:
        if repeat == 5:
            return reply
        else:
            reply = chating(tmp_input, dev_data, neg, num_examples,model, task, repeat+1)
    print(input_text)
    print('model\'s answer:'+reply)
    return reply

def load_data(path):
    return json.load(open(path,'r'))

def prepare_examples(dev_data,neg,nums,task):
    tmp = ''
    if task == 'TFQ' or task == 'TFQ_options':
        for one in dev_data[:nums]:
            ques = one[1] if not neg else one[2]
            ans = one[3] if not neg else one[4]
            
            if isinstance(ans,list):
                ans = ['true' if w == 'T' else 'false' for w in ans]
                ans = '、'.join(ans)
            else:
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
    model = args.model
    cnt = -1
    for typ in ['MCQ','MAQ','TFQ','RQ']:
        dev_path = os.path.join('medqa/{}'.format(typ),'dev.json')
        path = os.path.join('medqa/{}'.format(typ), 'test.json')
        dev_data = load_data(dev_path)

        data = load_data(path)
        out_data = []
        
        if not os.path.exists('results/medqa/{}'.format(typ)):
            os.makedirs('results/medqa/{}'.format(typ))
        f = open('results/medqa/{1}/{0}_{1}_results.json'.format(model,typ),'a')

        for item in tqdm(data):
            cnt += 1
            if cnt < args.start:
                continue
            if typ == 'TFQ' or typ=='TFQ_options':
                idx = item[0]
                ques = item[1]
                neg_ques = item[2]
                ques_2 = item[3]
                neg_ques_2 = item[4]
                input_text = ques + '\nAnswer: '
                neg_input_text = neg_ques + '\nAnswer: '
                input_text_2 = ques_2 + '\nAnswer: '
                neg_input_text_2 = neg_ques_2 + '\nAnswer: '
                reply = chating(input_text, dev_data, False, args.ntrain, model, typ)
                neg_reply = chating(neg_input_text, dev_data, True, args.ntrain, model, typ)
                reply_2 = chating(input_text_2, dev_data, False, args.ntrain, model, typ)
                neg_reply_2 = chating(neg_input_text_2, dev_data, True, args.ntrain, model, typ)
                out_data = item + [reply, neg_reply, reply_2, neg_reply_2]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
            elif typ == 'MAQ':
                idx = item[0]
                ques = item[1]
                neg_ques = item[2]
                input_text = ques + '\nAnswer: '
                neg_input_text = neg_ques + '\nAnswer: '
                reply = chating(input_text, dev_data, False, args.ntrain, model, typ)
                neg_reply = chating(neg_input_text, dev_data, True, args.ntrain, model, typ)
                out_data = item + [reply, neg_reply]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
            elif typ == 'MCQ' or typ == 'FIB':
                idx = item[0]
                ques = item[1]
                input_text = ques + '\nAnswer: '
                reply = chating(input_text, dev_data, False, args.ntrain, model, typ)
                out_data = item + [reply]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
            elif typ == 'RQ':
                idx = item[0]
                
                ques = item[1]
                ques_2 = item[2]
                input_text = ques + '\nAnswer: '
                input_text_2 = ques_2 + '\nAnswer: '
                reply = chating(input_text, dev_data, False, args.ntrain, model, typ)
                reply_2 = chating(input_text_2, dev_data, False, args.ntrain, model, typ)
                out_data = item + [reply, reply_2]
                f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
                f.flush()
        f.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    main(args)

