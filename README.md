# 👩‍🏫👩‍🏫[“ Leeing ”](https://github.com/lightxxxshadow/Leeing--NewsGPT) —— NewBeing 的简略模仿👩‍🏫👩‍🏫

<p align="center">
    <img src="./封面.jpg" width=300px/>
</p>

**这是一个低配版本的Newbeing,它与网络链接，可以通过网络接口获取最新的每日新闻，并借助这些新闻来回答你的问题。**
* 设计并训练了基于Bert的token级别的关键词抽取模型。
* 利用新闻访谈数据微调ChatGLM以获得更好的效果。
* 利用两个新闻接口分别提取最新消息与最匹配消息，并基于Dense Retrieval算法选择最相关新闻作为输入。
* 实现了网页版的demo版本测试。

<p align="center">
    <img src="./展示.png" width=2000px/>
</p>

## 最近更新
- 👏🏻  2023.06.20: Leeing-v1版本发布.实现了基本全部功能，支持网页版交互与命令行交互两种模式。

## 使用方法
* 实验环境

python 3.8

GeForce RTX 3090 ，24g显存

* 克隆本项目
```bash
cd ~
https://github.com/lightxxxshadow/Leeing--NewsGPT.git
```

* 安装依赖    
```python
pip install -r requirements.txt
```
需要注意的是torch的版本需要根据你的服务器实际的cuda版本选择，详情参考[pytorch安装指南](https://pytorch.org/get-started/previous-versions/)
```python
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

* 模型下载
- 可以选择利用自己的或者下面给出的数据进行自己的模型的训练，也可以直接用我训练好的模型测试效果。


- [在这里](https://drive.google.com/file/d/1T8H9lq2L2pnzqMSKiwEnR4a7O2PWlXmd/view?usp=sharing)下载用于关键词提取的模型，将其放在./Keyword_Extractor路径下即可，微调时GPU要求最低显存3g左右。

<p align="center">
    <img src="./key_extra训练.png" width=800px/>
</p>

- [在这里](https://drive.google.com/file/d/1rRfy7sc2tJ2fyjUWxgnrrU6gi-zyqwwt/view?usp=sharing)下载基于下面的数据微调后的ChatGLM模型的prefix部分参数模型，将整个output文件夹放置在./chatglm_ptuning路径下，
将其放在./Keyword_Extractor路径下即可，进行微调时显存需要17g左右。

 
<p align="center">
    <img src="./chatglm训练.png" width=800px/>
</p>

 
- 注意同时需要下载清华大学发布的[ChatGLM模型](https://github.com/THUDM/ChatGLM-6B)，我这里选择的是最小的4int级别量化版本，也可以不微调直接使用，显存消耗6-7g，将下载后的整个chatglm-6b-int4文件姐放在与本README同级目录下，
注意修改代码中本地的模型路径！！！

* 数据下载
- 除了直接下载训练好的模型，可以通过下面的方法自己训练自己的模型。


- [在这里](https://drive.google.com/file/d/12chZA87VUviFyOh1qWs8DI33hbjKsKiv/view)下载基于Bert的关键词提取模型的训练数据，格式压缩的jsonl文件，
解压后将其放在./Keyword_Extractor目录下，运行data_process.py即可获得用于训练的train.json数据，处理并清洗后大约有3万多条数据。


- [在这里](https://drive.google.com/file/d/1ZAKZM1cGhEw2A4_n4bGGMYyF8iPjLZni/view)下载用于微调ChatGLM模型的数据，该数据为英文的新闻访谈类多轮对话数据十分适合本任务，
解压后将其放在./chatglm_ptuning目录下，分别运行data_process.py中两个函数：process1和process2，即可获得用于训练的train.json数据。


- 本次实验所给出模型由于时间以及算力原因，Chat_GLM模型的微调只是用了约两万轮的带上下文信息的训练数据，不到总数据量的1%。


* 启动服务   

本项目提供了[demo_show.py](./demo_show.py)作为Leeing模型的使用示例，找到该文件路径下，通过以下命令即可开启服务：
```bash
streamlit run demo_show.py
```
特别地，在网页端运行时需要至少满足7个g左右显存，不然可能无法运行。运行时由于所使用新闻api为境外网址，需要翻墙，开启海外服务。
运行前特别注意安装包：
```python
pip install streamlit
pip install streamlit_chat
```
<p align="center">
    <img src="./demo.png" width=1000px/>
</p>

## 参考仓库
```bib
@misc{chen2023soulchat,
      title={灵心健康大模型SoulChat：通过长文本咨询指令与多轮共情对话数据集的混合微调，提升大模型的“共情”能力},
      author={Yirong Chen, Xiaofen Xing, Zhenyu Wang, Xiangmin Xu},
      year={2023},
      month = {6},
      version = {1.0},
      url = {https://github.com/scutcyr/SoulChat}
}
```
```bib
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```
```bib
@misc{Rahul1582，
      title={BERT-Keyword-Extractor},
      author={BERT-Keyword-Extractor, Xiaofen Xing, Zhenyu Wang, Xiangmin Xu},
      year={2019},
      month = {5},
      url = {https://github.com/ibatra/BERT-Keyword-Extractor}
}
```
