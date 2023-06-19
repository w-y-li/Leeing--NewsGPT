import json
def porcess1():
    with open('news_dialogue.json', 'r') as f:
        js = json.load(f)  # 此时a是一个字典对象
        data=[]
        for i in range(len(js)):
            for j in range(1,len(js[i]['utt'])-2,2):
                dict_data={}
                dict_data['user1'] = js[i]['utt'][j]
                dict_data['user2'] = js[i]['utt'][j+1]
                if j==1:
                    dict_data['history'] = []
                else:
                    dict_data['history'] = [js[i]['utt'][j-2],js[i]['utt'][j-1]]
                data.append(dict_data)

        with open('finetune_data.json', 'w') as f2:
            json.dump(data, f2)

def process2():
    with open('finetune_data.json', 'r') as f:
        js = json.load(f)
        with open('train.json', 'w') as f2:
            with open('eval.json', 'w') as f3:
                json.dump(js[:1000], f3)
                json.dump(js[1001:21001], f2)