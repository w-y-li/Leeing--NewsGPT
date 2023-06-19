import json
with open('data.jsonl', 'r') as f1:
    # 遍历文件的每一行
    data_all=[]
    for line in f1:
        # 解析每一行为一个Python字典
        tmp={}
        data = json.loads(line)
        tmp['key_w']=data['keyword']
        tmp['content'] = data['abstract']
        data_all.append(tmp)
    with open('train.json', 'w') as f2:
        with open('test.json', 'w') as f3:
            json.dump(data_all[:500], f3)
            json.dump(data_all, f2)
