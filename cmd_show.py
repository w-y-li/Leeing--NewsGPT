from urllib.parse import quote
from newsapi import NewsApiClient
import json
import urllib.request
from chatglm_ptuning import chat
from Keyword_Extractor import key_extract
import argparse
from Keyword_Extractor import abbreviation
import Dense_Retriever

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='model.pt',help='path to load model')
parser.add_argument('--News_api_key', type=str, default='aeb359474c014f749e44d03010bcf0f9')
parser.add_argument('--Gnews_api_key', type=str, default="6788e612d01e54b3e2e85cd25c49b308")
parser.add_argument('--News_data_from', type=str, default='2023-06-01')
parser.add_argument('--stopwords', type=str, default='./Keyword_Extractor/stopwords.txt')
parser.add_argument('--key_extract_model', type=str, default='./Keyword_Extractor/model/key_extra_model.pt')

args = parser.parse_args()


# 设置你的API key
newsapi1 = NewsApiClient(api_key=args.News_api_key)
newsapi2 = args.Gnews_api_key

def get_news1(key_news):
    top_headlines = newsapi1.get_everything(q=key_news,
                                      #sources='bbc-news,the-verge',
                                      #domains='bbc.co.uk,techcrunch.com',
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


def main():
    history = [('',''),('','')]
    turns=0
    new_time_flag=False
    DR_model,DR_tokenizer=Dense_Retriever.prepare()
    chatmodel,chat_tokenizer=chat.prepare()
    stopwords = key_extract.stop_w(args.stopwords)
    key_model,key_tokenizer=key_extract.prepare(args.key_extract_model)

    while (True):
        user_text = input('User>>')
        if(user_text=='Z'):
            print("See you next time ~")
            break

        text_for_key=key_extract.preprocess(user_text,stopwords,abbreviation.limits)
        key_list=key_extract.keywordextract(key_tokenizer,text_for_key,key_model)

        if key_list==-1:
            response, his = chat.chat(chat_tokenizer, user_text, [history[(turns + 1) % 2]], chatmodel)
            history[turns % 2] = his[1]

            print("NewsChat_Bot>>%s" % response)
            print("(Print Z to break)")
            print("End turns %d --------------------------------------------------------------------------------------------------------------------------------------------------------------------------" % turns)
            turns += 1
            continue

        key_news_rele = get_news1(' '.join(key_list))
        key_news_time = get_news2(' '.join(key_list))
        if key_news_time ==[]:
            new_time_flag=True

        news_fin ,url= get_final_news(new_time_flag,key_news_rele,key_news_time,user_text,DR_tokenizer, DR_model)
        prompt = f"Please read the following news and remember it:[{news_fin}] based on the news ,talk about the question:[{user_text}]，first retell the news and then give your answer to the question activly and vivdly."
        response, his = chat.chat(chat_tokenizer, prompt, [history[(turns+1)%2]],chatmodel)
        history[turns % 2][0] = user_text
        history[turns % 2][1] = his[1][1]

        print("<^-^> NewsChat_Bot>>%s"%response)
        print('<^-^> Using news from : ',url)
        print("(Print Z to break)")
        print("<^-^> End turns %d --------------------------------------------------------------------------------------------------------------------------------------------------------------------------"%turns)
        turns+=1

def get_final_news(new_time_flag,news1,news2,user,tokenizer,model):
    news_final1,news_final2=Dense_Retriever.main(new_time_flag,news1,news2,user,tokenizer,model)

    if not new_time_flag:
        news_fin=news_final1[0].split('<^-^>')[0]+news_final2[0].split('<^-^>')[0]
        url='['+news_final1[0].split('<^-^>')[1]+'+]'+'['+news_final2[0].split('<^-^>')[1]+'+]'
    else:
        news_fin = news_final1[0].split('<^-^>')[0]
        url = '['+news_final1[0].split('<^-^>')[1]+'+]'
    return news_fin,url

if __name__=="__main__":
    main()
