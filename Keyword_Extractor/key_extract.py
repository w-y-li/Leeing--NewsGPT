import string
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer


def prepare(path):
    model = torch.load(path)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return model,tokenizer

def stop_w(path):
    with open(path, 'r') as f:
        f=f.readlines()
        s_words=[]
        for i in range(len(f)):
            f[i]=f[i].strip()
            s_words.append(f[i].lower())
        return s_words

def preprocess(text,stopwords,limits):
    text=text.lower()
    limits_lower = {k.lower(): v for k, v in limits.items()}
    punctuation_trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text_sp=text.split()
    for key, value in limits_lower.items():
        key_lower = key.lower()
        if key_lower in text_sp:
            text_sp[text_sp.index(key_lower)] = value.lower()
    text = text.translate(punctuation_trans)
    text_sp = text.split()
    tmp=[]
    for i in range(len(text_sp)):
        if text_sp[i] not in stopwords:
            tmp.append(text_sp[i])
    text = ' '.join(tmp)
    return text

def keywordextract(tokenizer,sentence,model):
    text = sentence
    tkns = tokenizer.tokenize(text)
    if len(tkns)==0:
        return -1
    indexed_tokens = tokenizer.convert_tokens_to_ids(tkns)
    segments_ids = [0] * len(tkns)
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    segments_tensors = torch.tensor([segments_ids]).cuda()
    logit = model(tokens_tensor, token_type_ids=None,
                                  attention_mask=segments_tensors)
    logit = logit.detach().cpu().numpy()[0]
    logit=list(logit)
    keys=[]
    for i in range(len(logit)):
        logit[i]=list(logit[i])
        keys.append(logit[i][0])
    keys_sort=sorted(keys,reverse=True)
    prediction=list(np.argmax(logit, axis=1))

    num_0=prediction.count(0)

    if num_0>3:
        for item in keys_sort[3:]:
            prediction[keys.index(item)] = 1

    i=0
    while num_0 < 3 and i<len(keys_sort):
        if keys_sort[i]<0.5:
            prediction[keys.index(keys_sort[i])]=0
            num_0+=1
        i+=1

    result=[]
    for i in range(len(prediction)):
        if prediction[i]==0:
            result.append(tkns[i])
    return result

# model,tokenizer=prepare('./model/model_v3.pt')
# text="Recently,China made a great progress in making America and Japan fight in a war,Many people even don't believe that,But according to Xi jinping,that's true."
# text=preprocess(text,stop_w('stopwords.txt'))
# print(keywordextract(tokenizer,text,model))