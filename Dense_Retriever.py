from transformers import AutoTokenizer, AutoModel
import torch
from annoy import AnnoyIndex

def prepare():
    # 加载预训练的BERT模型和对应的分词器
    model = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return model,tokenizer

def main(new_time_flag,news1,news2,user,tokenizer,model):
    query = user
    inputs = tokenizer([query], return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()

    new_result1 = {}
    new_result2 = {}
    if not new_time_flag:
        news_list2 = news2
        inputs2 = tokenizer(news_list2, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs2 = {name: tensor.to(model.device) for name, tensor in inputs2.items()}

        with torch.no_grad():
            news_embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1).cpu().numpy()

        f2 = news_embeddings2.shape[1]
        index2 = AnnoyIndex(f2, 'angular')  # Length of item vector that will be indexed

        for i in range(len(news_list2)):
            index2.add_item(i, news_embeddings2[i])

        index2.build(10)
        I2 = index2.get_nns_by_vector(query_embedding[0], 2)

        for i in range(len(I2)):
            new_result2[i] = news_list2[I2[i]]

    news_list1 = news1
    inputs1 = tokenizer(news_list1, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs1 = {name: tensor.to(model.device) for name, tensor in inputs1.items()}

    with torch.no_grad():
        news_embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1).cpu().numpy()

    f1 = news_embeddings1.shape[1]
    index1 = AnnoyIndex(f1, 'angular')  # Length of item vector that will be indexed

    for i in range(len(news_list1)):
        index1.add_item(i, news_embeddings1[i])

    index1.build(10)
    I1 = index1.get_nns_by_vector(query_embedding[0], 2)

    for i in range(len(I1)):
        new_result1[i] = news_list1[I1[i]]

    return new_result1,new_result2