'''
    recognize the type of medical exam questions, and rewrite them into new questions (MAQ, TFQ, RQ)
'''
import json
import re
from collections import Counter

prefix = r"(?:\. *([\w ’\'\-,]*?))?(what|which of the following)(\s[\w\s’\'\-]*?)(is|are|was|should be|should also be|could have|will be|would have|could be|would be|would|could) "
prefix_2 = r"(?:\. *([\w ’\'\-,]*?))?(what|which of the following)([\w\s’\'\-]*?) "
patterns = {
        r"(characteristic of )?the most likely diagnosis(.*?)(?:\?|\:)": ['Which of the followings are the most likely diagnoses? There may be one or more correct choices.','Which of the followings are less likely to be the diagnosis? There may be one or more correct choices.','"{} is likely to be the diagnosis in this case", is the statement above true or false? Please answer true/false.','"{} is unlikely to be the diagnosis in this case", is the statement above true or false? Please answer true/false.'],
        r"the (.*?)mechanism(.*?)(?:\?|\:)":['Which of the followings are the most likely underlying mechanisms of this patient\'s condition? There may be one or more correct choices.','Which of the followings are less likely to be the underlying mechanism of this patient\'s condition? There may be one or more correct choices.','"{} is likely to be the underlying mechanism of this patient\'s condition", is the statement above true or false? Please answer true/false.','"{} is unlikely to be the underlying mechanism of this patient\'s condition", is the statement above true or false? Please answer true/false.'],
        r"the (.*?)explanation(.*?)(?:\?|\:)":['Which of the followings are the most likely underlying mechanisms of this patient\'s condition? There may be one or more correct choices.','Which of the followings are less likely to be the underlying mechanism of this patient\'s condition? There may be one or more correct choices.','"{} is likely to be the underlying mechanism of this patient\'s condition", is the statement above true or false? Please answer true/false.','"{} is unlikely to be the underlying mechanism of this patient\'s condition", is the statement above true or false? Please answer true/false.'],
        r"the most specific etiology(.*?)(?:\?|\:)":[],
        r"the (.*?)(immediate|initial|first|next|long-term) step in ((?:the|his|her|this patient’s)\s)?(management|therapy|treatment|evaluation|action)(.*?)(?:\?|\:)":['Which of the followings are the most likely next best steps in management? There may be one or more correct choices.','Which of the followings are less likely to be the next best step in management? There may be one or more correct choices.','"{} is likely to be the next best step in management", is the statement above true or false? Please answer true/false.','"{} is unlikely to be the next best step in management", is the statement above true or false? Please answer true/false.'],
        r"the (.*?)(diagnostic test|diagnostic step|step in diagnosis)(.*?)(?:\?|\:)":[],
        r"the (.*?)next step(.*?)(?:\?|\:)":[],
        r"the([\w\s’\'\-]*?) (management|therapy|treatment|intervention|action|recommendation)([\w\s’\'\-]*?)(?:\?|\:)":[],
        r"the (immediate|initial|first|next|long-term) ([\w\s’\'\-]*?)step(.*?)(?:\?|\:)":['Which of the followings are the most likely next best steps in management? There may be one or more correct choices.','Which of the followings are less likely to be the next best step in management? There may be one or more correct choices.','"{} is likely to be the next best step in management", is the statement above true or false? Please answer true/false.','"{} is unlikely to be the next best step in management", is the statement above true or false? Please answer true/false.'],
        r"the([\w\s’\'\-]*?) etiology (.*?)(?:\?|\:)":['Which of the followings are the most likely etiologies of this patient\'s condition? There may be one or more correct choices.','Which of the followings are less likely to be the etiology of this patient\'s condition? There may be one or more correct choices.','"{} is likely to be the etiology of this patient\'s condition", is the statement above true or false? Please answer true/false.','"{} is unlikely to be the etiology of this patient\'s condition", is the statement above true or false? Please answer true/false.'],
        r"(true of )?the (.*?)cause (of|for) (.*?)(?:\?|\:)":['Which of the followings are the most likely causes of the patient\'s condition? There may be one or more correct choices.','Which of the followings are less likely to be the causes of the patient\'s condition? There may be one or more correct choices.','"{} is likely to be the cause of the patient\'s condition", is the statement above true or false? Please answer true/false.','"{} is unlikely to be the cause of the patient\'s condition", is the statement above true or false? Please answer true/false.'],
        r"(a )?(characteristic of )?the (.*?) (that is )?(causing|responsible for) (.*?)(?:\?|\:)":['Which of the followings are the most likely causes of the patient\'s condition? There may be one or more correct choices.','Which of the followings are less likely to be the causes of the patient\'s condition? There may be one or more correct choices.','"{} is likely to be the cause of the patient\'s condition", is the statement above true or false? Please answer true/false.','"{} is unlikely to be the cause of the patient\'s condition", is the statement above true or false? Please answer true/false.'],
        r"(most likely )?(causing|responsible for) (.*?)(?:\?|\:)":['Which of the followings are the most likely causes of the patient\'s condition? There may be one or more correct choices.','Which of the followings are less likely to be the causes of the patient\'s condition? There may be one or more correct choices.','"{} is likely to be the cause of the patient\'s condition", is the statement above true or false? Please answer true/false.','"{} is unlikely to be the cause of the patient\'s condition", is the statement above true or false? Please answer true/false.'],
        r"the (.*?)(predisposing|inciting|precipitating) factor (.*?)(?:\?|\:)":[],
        r"(a|the) (.*?) factor (.*?)(?:\?|\:)":[],
        r"the (.*?)(next|initial) (.*?)pharmacotherapy(.*?)(?:\?|\:)":[],
        r"the (.*?)(pharmacotherapy|drug)(.*?)(?:\?|\:)":[],
        r"(.*?)confirm (the|your) diagnosis(.*?)(?:\?|\:)":[],
        r"(.*?)(associated|consistent) with(.*?)(?:\?|\:)":[],
        r"(.*?)contributed to(.*?)(?:\?|\:)":[],
        r"the([\w\s’\'\-]+?)finding (.*?)(?:\?|\:)":[],
        r"the([\w\s’\'\-]+?) cause(.*?)(?:\?|\:)":[],
        r"(.*?)(seen|found|observed|expected|present )([\w\s’\'\-]*?)(?:\?|\:)":[],
        r"([\w\s’\'\-]*?)true( about)?(.*?)(?:\?|\:)":[],
        r"([\w\s’\'\-]*?)involved in(.*?)(?:\?|\:)":[],
        r"([\w\s’\'\-]*?)(prevented|reducing|improve) (.*?)(?:\?|\:)":[],
        r"([\w\s’\'\-]*?)(administered|added to|used to treat|prescribed to)(.*?)(?:\?|\:)":[],
        r"([\w\s’\'\-]*?)affected by(.*?)(?:\?|\:)":[],
        r"([\w\s’\'\-]+?)(management|therapy|treatment|intervention)([\w\s’\'\-]*?)(?:\?|\:)":[],
        # r"(.*?)to evaluate(.*?)(?:\?|\:)":[],
    }
patterns_2 = {
        r"(best )?(represents|describes|explains|explain|accounts for) (.*?)(?:\?|\:)":[],
        r"tests (is|are) (required to|indicated to|most likely to) (investigate|determine|elucidate) (.*?)(?:\?|\:)":[],
        r"contributed to (.*?)(?:\?|\:)":[],

    }
def recognize_and_rewrite(ques):
    '''
        recognize the question and rewrite it into multiple questions
    '''
    # ques = ques.lower()
    def rewrite(ans_list, pattern_idx, target_idx):
        neg_word_dict = {
            'is':'is not',
            'are':'are not',
            'were':'were not',
            'was':'was not',
            'should be':'should not be',
            'should also be':'should not be',
            'could have':'could not have',
            'will be':'will not be',
            'would have':'would not have',
            'could be':'could not be',
            'would be':'would not be',
            'would':'would not',
            'could':'could not'
        }
        plu_dict = {k:k for k in neg_word_dict}
        plu_dict['is'] = 'are'
        plu_dict['was'] = 'were'
        tails_list = [
            r'{}the most likely diagnosis{}',
            r'the {}mechanism{}',
            r"the {}explanation{}",
            r"the most specific etiology{}",
            r"the {}{} step in {}{}{}",
            r"the {}{}{}",
            r"the {}next step{}",
            r"the{} {}{}",
            r"the {} {}step{}",
            r"the{} etiology {}",
            r"{}the {}cause {} {}",
            r"{}{}the {} {}{} {}",
            r"{}{} {}",
            r"the {}{} factor {}",
            r"{} {} factor {}",
            r"the {}{} {}pharmacotherapy{}",
            r"the {}{}{}",
            r"{}confirm {} diagnosis{}",
            r"{}{} with{}",
            r"{}contributed to{}",
            r"the{}finding {}",
            r"the{} cause{}",
            r"{}{}{}",
            r"{}true{}{}",
            r"{}involved in{}",
            r"{}{} {}",
            r"{}{}{}",
            r"{}affected by{}",
            r"{}{}{}",
            r"{}{} {}",
            r"tests {} {} {}",
            r"contributed to {}",
        ]
        neg_tails_list = [
            r"do not {}{} {}",
            r"tests {} not {} {} {}",
            r"do not contributed to {}",
        ]
        new_list = []
        for one in range(len(ans_list.regs)):
            if ans_list[one]:
                new_list.append(ans_list[one])
            else:
                new_list.append('')
        ans_list = new_list
        if pattern_idx == 0:
            plu_word = plu_dict[ans_list[4]]
            neg_plu_word = neg_word_dict[plu_word]
            maq = '{}{}{}{} '.format(ans_list[1], ans_list[2], ans_list[3], plu_word)
            neg_maq = '{}{}{}{} '.format(ans_list[1], ans_list[2], ans_list[3], neg_plu_word)
            tfq = '{}[option_text] {} '.format(ans_list[1], ans_list[4])
            neg_tfq = '{}[option_text] {} '.format(ans_list[1], neg_word_dict[ans_list[4]])
            fib = 'Fill in the blank space below with the appropriate content: {}___ {} '.format(ans_list[1], ans_list[4])
        else:
            maq = '{}{}{} '.format(ans_list[1], ans_list[2], ans_list[3])
            neg_maq = '{}{}{} '.format(ans_list[1], ans_list[2], ans_list[3])
            tfq = '{}[option_text] '.format(ans_list[1])
            neg_tfq = '{}[option_text] '.format(ans_list[1])
            fib = 'Fill in the blank space below with the appropriate content: {}___ '.format(ans_list[1])
        tail = tails_list[len(patterns)*pattern_idx+target_idx].format(*ans_list[4:]) if pattern_idx == 1 else tails_list[len(patterns)*pattern_idx+target_idx].format(*ans_list[5:])
        neg_tail = neg_tails_list[target_idx].format(*ans_list[4:]) if pattern_idx == 1 else None
        maq = maq + tail + '? There may be one or more correct choices.'
        neg_maq = neg_maq + tail +'? There may be one or more correct choices.' if pattern_idx == 0 else neg_maq + neg_tail +'? There may be one or more correct choices.'
        tfq_1 = 'Statement: "'+tfq + tail+'.", is the statement above true or false? Please answer true/false.'
        neg_tfq_1 = 'Statement: "'+neg_tfq+tail+'.", is the statement above true or false? Please answer true/false.' if pattern_idx == 0 else 'Statement: "'+neg_tfq+neg_tail+'.", is the statement above true or false? Please answer true/false.'
        tfq_2 = 'Statement: "'+tfq + tail+' comparing with other four options: {}, {}, {}, and {}.", is the statement above true or false? Please answer true/false.'
        neg_tfq_2 = 'Statement: "'+neg_tfq + tail+' comparing with other four options: {}, {}, {}, and {}.", is the statement above true or false? Please answer true/false.' if pattern_idx == 0 else 'Statement: "'+neg_tfq+neg_tail+' comparing with other four options: {}, {}, {}, and {}.", is the statement above true or false? Please answer true/false.'
        
        fib = fib + tail + '.'
        
        
        return maq, neg_maq, tfq_1, neg_tfq_1, tfq_2, neg_tfq_2, fib
    
    # patterns = {prefix+p:v for p,v in raw_patterns.items()}
    # patterns_2 = {prefix_2+p:v for p,v in raw_patterns.items()}
    ans_list = None
    ans = []
    target_p = None
    final_phrase = None
    target_idx = -1
    pattern_idx = -1
    ques_list = None
    for i, p in enumerate(patterns):
        if not ans_list:
            ans_list=re.finditer(prefix+p,ques,re.DOTALL|re.I)
            try:
                ans_list = list(ans_list)
                # if len(ans_list) > 0 and i==21:
                #     print('y')
                ans_list = ans_list[-1]
            except:
                ans_list = None
            target_idx = i
            pattern_idx = 0
            # if i == 4 and ans_list:
            #     print(ans_list[0])
        else:
            break
    if not ans_list:
        for i, p in enumerate(patterns_2):
            if not ans_list:
                ans_list = re.finditer(prefix_2+p,ques,re.DOTALL|re.I)
                try:
                    ans_list = list(ans_list)[-1]
                except:
                    ans_list = None
                target_p = p
                target_idx = i
                pattern_idx = 1
            else:
                break

    ques_list=re.search(prefix,ques,re.DOTALL|re.I)
        # if i == 4 and ans_list:
    if not ques_list:
        ques_list=re.search(prefix_2,ques,re.DOTALL|re.I)

    # if ques_list and not ans_list:
    #     print(ques[ques.index(ques_list[0]):])
    maq, neg_maq, tfq, neg_tfq, tfq_2, neg_tfq_2, fib = None, None, None,None,None,None,None
    if ans_list:
        # for i in range(len(ans_list.regs)):
        #     print(ans_list[i], end='')
        # print('\n')
        ans = ans_list[0]
        maq, neg_maq, tfq, neg_tfq, tfq_2, neg_tfq_2, fib = rewrite(ans_list, pattern_idx, target_idx)

        # final_phrase = ans_list[1]
    if not target_p:
        out = []
    elif pattern_idx == 0:
        out = patterns[target_p] 
    else:
        out = patterns_2[target_p] 
    return pattern_idx, target_idx, target_p, ans, out, ans_list, maq, neg_maq, tfq, neg_tfq, tfq_2, neg_tfq_2, fib

data = []
with open('medqa/questions/US/test.jsonl','r') as f:
    for line in f:
        data.append(json.loads(line.strip()))
hit = 0
ques_set = set()
counter = []
ques_pattern = r"(?:what|which of the following).*"
unmatched_ques = set()
counter_2 = []
outf = open('rewrite.jsonl','w')
for item in data:
    ques = item['question']
    options = item['options']
    answer = item['answer_idx']
    pattern_idx, target_idx, target_p, ans_list, out, matchobj, maq, neg_maq, tfq, neg_tfq, tfq_2, neg_tfq_2, fib = recognize_and_rewrite(ques)
    
    if len(ans_list) > 0:
        hit += 1
        
        # print(ques)
        # for one in options:
        #     print('{}: {}\t'.format(one, options[one]),end='')
        # print('\n')
        ques_set.add((pattern_idx, target_idx, target_p, ans_list))
        counter.append((pattern_idx, target_idx))
        # outf.write('{}\t{}\n'.format(ans_list, maq))
        # outf.write('{}\t{}\n'.format(ans_list, neg_maq))
        # outf.write('{}\t{}\n'.format(ans_list, tfq))

        maq_ques = ques.replace(ans_list, '. '+maq)
        neg_maq_ques = ques.replace(ans_list, '. '+neg_maq)
        tfq_ques = ques.replace(ans_list, '. '+tfq)
        neg_tfq_ques = ques.replace(ans_list, '. '+neg_tfq)
        tfq_2_ques = ques.replace(ans_list, '. '+tfq_2)
        neg_tfq_2_ques = ques.replace(ans_list, '. '+neg_tfq_2)
        assert 'Statement' in tfq_ques
        assert 'Statement' in neg_tfq_ques
        assert 'Statement' in tfq_2_ques
        assert 'Statement' in neg_tfq_2_ques
        fib_ques = ques.replace(ans_list, '. '+fib)

        out = {
            'ques':ques,
            'maq_ques':maq_ques,
            'neg_maq_ques':neg_maq_ques,
            'tfq_ques':tfq_ques,
            'neg_tfq_ques':neg_tfq_ques,
            'tfq_2_ques':tfq_2_ques,
            'neg_tfq_2_ques':neg_tfq_2_ques,
            'fib_ques':fib_ques,
            'options':options,
            'answer':answer
        }
        outf.write(json.dumps(out,ensure_ascii=False)+'\n')
        outf.flush()
print(hit)
outf.close()