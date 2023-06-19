import os
import torch
import streamlit as st
from streamlit_chat import message
from urllib.parse import quote
from newsapi import NewsApiClient
import json
import urllib.request
from chatglm_ptuning import chat
from Keyword_Extractor import key_extract
import argparse
from Keyword_Extractor import abbreviation
import Dense_Retriever

# 指定显卡进行推理
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='model.pt',help='path to load model')
parser.add_argument('--News_api_key', type=str, default='aeb359474c014f749e44d03010bcf0f9')
parser.add_argument('--Gnews_api_key', type=str, default="6788e612d01e54b3e2e85cd25c49b308")
parser.add_argument('--News_data_from', type=str, default='2023-06-01')
parser.add_argument('--stopwords', type=str, default='./Keyword_Extractor/stopwords.txt')
parser.add_argument('--key_extract_model', type=str, default='./Keyword_Extractor/key_extra_model.pt')

args = parser.parse_args()

# 设置你的API key
newsapi1 = NewsApiClient(api_key=args.News_api_key)
newsapi2 = args.Gnews_api_key

def answer(history,user_text,DR_model,DR_tokenizer,chatmodel,chat_tokenizer,stopwords,key_model,key_tokenizer):
    text_for_key = key_extract.preprocess(user_text, stopwords, abbreviation.limits)
    key_list = key_extract.keywordextract(key_tokenizer, text_for_key, key_model)

    if key_list == -1:
        response, his = chat.chat(chat_tokenizer, user_text, history, chatmodel)

        return response,-1

    key_news_rele = get_news1(' '.join(key_list))
    key_news_time = get_news2(' '.join(key_list))
    new_time_flag=False
    if key_news_time == []:
        new_time_flag = True
    news_fin, url = get_final_news(new_time_flag, key_news_rele, key_news_time, user_text, DR_tokenizer, DR_model)
    prompt = f"Please read the following news and remember it:[{news_fin}] based on the news ,talk about the question:[{user_text}]，first retell the news and then give your answer to the question activly and vivdly."
    response, his = chat.chat(chat_tokenizer, prompt, history, chatmodel)

    return response,url

def get_news1(key_news):
    top_headlines = newsapi1.get_everything(q=key_news,
                                      from_param=args.News_data_from,
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=10,
                                      qintitle='title')
    key_news = []

    for i in range(int(len(top_headlines['articles']))):
        key_news.append(top_headlines['articles'][i]['title'] + ','+ top_headlines['articles'][i]['description'] + '<^-^>'+top_headlines['articles'][i]['url'])

    return key_news


def get_news2(key_news):
    key_news= quote(key_news)
    url = f"https://gnews.io/api/v4/search?q={key_news}&lang=en&max=10&apikey={newsapi2}&sortby=publishedAt"
    articles=[]
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = data["articles"]
    except:
        pass
    key_news = []

    for i in range(len(articles)):
        key_news.append(articles[i]['title']+','+articles[i]['description']+ '<^-^>'+articles[i]['url'])

    return key_news

def get_final_news(new_time_flag,news1,news2,user,tokenizer,model):
    news_final1,news_final2=Dense_Retriever.main(new_time_flag,news1,news2,user,tokenizer,model)

    if not new_time_flag:
        news_fin=news_final1[0].split('<^-^>')[0]+news_final2[0].split('<^-^>')[0]
        url='['+news_final1[0].split('<^-^>')[1]+'+]'+'['+news_final2[0].split('<^-^>')[1]+'+]'
    else:
        news_fin = news_final1[0].split('<^-^>')[0]
        url = '['+news_final1[0].split('<^-^>')[1]+'+]'
    return news_fin,url

st.set_page_config(
    page_title="Leeing-v1~~",
    page_icon="👩‍🏫",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """     
-   版本：👩‍🏫Leeing-v1~~
-   电话：13598800191
-   作者：李文豫
-   简介：这是一个联网的必应的一个模仿尝试版本，它可以看到最新的新闻，可以用新闻的知识来回答
	    """
    }
)

st.header("👩‍🏫 <^-^> Leeing-v1 <^-^> ")

with st.expander("ℹ️ -  <^-^> 关于我们", expanded=False):
    st.write(
        """     
-    <^-^> vision：👩‍🏫Leeing-v1~~
-    <^-^> phone：13598800191
-    <^-^> coder：李文豫
-    <^-^> <^-^> 这是一个联网的必应的一个模仿尝试版本，它可以看到最新的新闻，可以用新闻的知识来回答你的问题~ <^-^> <^-^> 
	    """
    )

@st.cache_resource
def pre_work():
    DR_model, DR_tokenizer = Dense_Retriever.prepare()
    chatmodel, chat_tokenizer = chat.prepare()
    stopwords = key_extract.stop_w(args.stopwords)
    key_model, key_tokenizer = key_extract.prepare(args.key_extract_model)
    print('Prepare work done!')
    return DR_model,DR_tokenizer,chatmodel,chat_tokenizer,stopwords,key_model,key_tokenizer

DR_model,DR_tokenizer,chatmodel,chat_tokenizer,stopwords,key_model,key_tokenizer=pre_work()

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'url' not in st.session_state:
    st.session_state['url'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

user_col, ensure_col = st.columns([5, 1])


def get_text():
    input_text = user_col.text_area(" <^-^> 请在下列文本框输入对话信息：", "", key="input",
                                    placeholder=" <^-^> 请输入您的对话信息，并且点击Ctrl+Enter(或者发送按钮)确认内容")
    if ensure_col.button("<-发送->", use_container_width=True):
        if input_text:
            return input_text
    else:
        if input_text:
            return input_text


user_input = get_text()

if 'id' not in st.session_state:
    if not os.path.exists("./history"):
        # 创建保存用户聊天记录的目录
        os.makedirs("./history")
    json_files = os.listdir("./history")
    id = len(json_files)
    st.session_state['id'] = id

if user_input:
    st.session_state.past.append(user_input)
    if len(st.session_state['generated']) ==0:
        history=[]
    else:
        history=[tuple([st.session_state['past'][-2],st.session_state['generated'][-1]])]
    output ,url= answer(history,st.session_state['past'][-1],DR_model,DR_tokenizer,chatmodel,chat_tokenizer,stopwords,key_model,key_tokenizer)
    st.session_state.generated.append(output)
    st.session_state.url.append(url)

    # 将对话历史保存成json文件
    dialog_history = {
        'user': st.session_state['past'],
        'bot': st.session_state["generated"],
        'news_url':st.session_state["url"]
    }
    with open(os.path.join("./history", str(st.session_state['id']) + '.json'), "w", encoding="utf-8") as f:
        json.dump(dialog_history, f, indent=4, ensure_ascii=False)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        if i == 0:
            message(
                " <^-^> 我是Leeing~~，欢迎找我聊聊新闻内容❤️，期待帮助到你！🤝🤝🤝" + "\n------------------\n ***请注意目前仅支持英文输入，请注意不要在未回复时再次发送消息到缓冲区！！！（右上角会显示是否在运行）***",
                key=str(i), avatar_style="avataaars", seed=5)
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed=26)
            if st.session_state["url"][i]!=-1:
                message(st.session_state["generated"][i]+'\n'+'URL : '+st.session_state["url"][i], key=str(i)+'_bot', avatar_style="avataaars", seed=5)
            else:
                message(st.session_state["generated"][i], key=str(i) + '_bot', avatar_style="avataaars", seed=5)
        else:
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed=26)
            if st.session_state["url"][i]!=-1:
                message(st.session_state["generated"][i]+'\n'+'URL : '+st.session_state["url"][i], key=str(i)+'_bot', avatar_style="avataaars", seed=5)
            else:
                message(st.session_state["generated"][i], key=str(i) + '_bot', avatar_style="avataaars", seed=5)
