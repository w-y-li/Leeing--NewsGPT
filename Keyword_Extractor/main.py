import json
import argparse
from nltk.tokenize import sent_tokenize
from abbreviation import limits
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertForTokenClassification
from keras_preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser(description='BERT Keyword Extraction Model')

parser.add_argument('--data', type=str, default=None,
                    help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=80, metavar='N',
                    help='sequence length')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

train_path = args.data

import string

def stop_w(path):
    with open(path, 'r') as f:
        f=f.readlines()
        s_words=[]
        for i in range(len(f)):
            f[i]=f[i].strip()
            if len(f[i])!=1:
                s_words.append(f[i].lower())
        return s_words

def convert(path):
    # Load stop words once before starting the loop
    stop_words = stop_w('stopwords.txt')

    # Prepare translation table for punctuation removal
    punctuation_trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    with open(path, 'r', encoding='utf-8') as f:
        js = json.load(f)
        tokens = []
        tmp_keys = []
        label = []
        data = []
        keys = []
        text=[]
        for i in range(len(js)):
            data.append(js[i]['content'][:512])
            tmp_keys.append(js[i]['key_w'].split(';'))

        for i in range(len(data)):
            sentence = sent_tokenize(data[i])
            for j in range(len(sentence)):
                tokens.append(sentence[j])
                keys.append(tmp_keys[i])

        limits_lower = {k.lower(): v for k, v in limits.items()}

        for i in trange(len(tokens), desc="processed_data"):
            token_sp=tokens[i].lower().split()
            for key, value in limits_lower.items():
                key_lower = key.lower()
                if key_lower in token_sp:
                    token_sp[token_sp.index(key_lower)]=value.lower()
                if key_lower in keys[i]:
                    keys[i][keys[i].index(key_lower)]=value.lower()
            tokens[i]=' '.join(token_sp)
            keys[i] = [k.lower().translate(punctuation_trans) for k in keys[i]]
            tokens[i] = tokens[i].translate(punctuation_trans)

            tokens[i] = ' '.join([kw for kw in tokens[i].split() if kw.lower() not in stop_words])
            for k in keys[i]:
                k = ' '.join([kw for kw in k.split() if kw not in stop_words])

            tokens_lower = tokens[i].lower().split()

            z = ['O'] * len(tokens_lower)
            for k in keys[i]:
                k_sp = k.split()
                ind = [idx for idx, tk in enumerate(tokens_lower) if tk in k_sp]
                for idx in ind:
                        z[idx] = 'B'

            if 'B' in z:
                label.append(z)
                text.append(tokens[i].lower())

        return text, label


path='train.json'
text, label = convert(path)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


MAX_LEN = args.seq_len
bs = args.batch_size
tag2idx = {'B': 0, 'O': 1}
tags_vals = ['B', 'O']


def prepare_dataloader(text,label):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)  #####################################

    tokenized_texts = [tokenizer.tokenize(par) for par in text]

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in label],
                         maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                         dtype="long", truncating="post")

    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=2023, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2023, test_size=0.1)

    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)  ##################################
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
    return train_dataloader,valid_dataloader

train_dataloader,valid_dataloader=prepare_dataloader(text,label)

def start_train(train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))
    model = model.cuda()

    FULL_FINETUNING = False
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=args.lr)

    epochs = args.epochs
    max_grad_norm = 1.0

    for i in range(epochs):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader)) as t:
            for step, batch in enumerate(train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # forward pass
                loss = model(b_input_ids, token_type_ids=None,
                             attention_mask=b_input_mask, labels=b_labels.long())
                # backward pass
                loss.requires_grad_(True)
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                # update parameters
                optimizer.step()
                model.zero_grad()
                t.set_description(desc="Training...")
                t.update(1)
        # print train loss per epoch
        print("\nTrain loss: {}".format(tr_loss / nb_tr_steps))

    torch.save(model, args.save)
start_train(train_dataloader)
def start_test(model,valid_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    true_labels = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels.long())
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

        label_ids = b_labels.to('cpu').numpy()
        true_labels.append(label_ids)
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    print("Validation loss: {}".format(eval_loss / nb_eval_steps))
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

#start_test(torch.load('model.pt'),valid_dataloader)