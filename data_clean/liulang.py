# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/9/8 8:13 下午
==================================="""


import pandas as pd

def liulang_process():
    data_path = '../data/liulangdiqiu/多轮对话生成_20220905.csv'
    all_data = {'question': [], 'answer':[], 'context':[]}
    conversation = []
    datas = pd.read_csv(data_path, index_col=0)
    historys = []
    for index, sentence in zip(datas.index, datas['dialog']):

        if "###分割线###" == sentence:
            continue
        if "name" == sentence:
            continue
        if pd.isna(index) or pd.isna(sentence):
            print(sentence)
            continue
        if int(index) == 0:
            if historys:
                conversation.append(historys)
            historys = [sentence]
        else:
            sentence = sentence.replace("@刘培强，", '').replace('@莫斯，', '').replace('@刘启，', '')\
                .replace('@韩朵朵，','').replace('@王磊，', '')
            sentence = sentence.strip('"\n').strip('\n')
            historys.append(sentence)

    for con in conversation:
        his = ''
        for i in range(0, len(con),2):
            ques = con[i].strip('"').strip('\n')
            if i+1 >= len(con):
                continue
            ans = con[i + 1].strip('"').strip('\n')

            all_data['question'].append(ques)
            all_data['answer'].append(ans)
            all_data['context'].append(his.strip('"'))
            his = his + ques + ans

    frames = pd.DataFrame(all_data)
    frames.to_csv('../data/liulangdiqiu/liulangdiqiu.csv', index=False)


def temp_func():
    with pd.read_csv('../data/liulangdiqiu/multiturn_qa_221125.csv', usecols=['question', 'answer', 'context'], chunksize=1000, sep=',') as reader:
        for i, chunk in enumerate(reader):
            querys = []
            answers = []
            contexts = []
            for query, ans, context in zip(chunk['question'], chunk['answer'], chunk['context']):
                if pd.isna(query) or pd.isna(ans):
                    continue

                context = '' if pd.isna(context) else context
                querys.append(query)
                answers.append(ans)
                contexts.append(context)

            print(f"Already loaded {i * 1000 +len(querys)} items data.")

if __name__ == '__main__':
    # liulang_process()
    temp_func()
