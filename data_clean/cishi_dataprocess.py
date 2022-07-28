# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/5/11 5:29 下午
==================================="""
import glob

import json
import re
import time

import pandas as pd
import requests
# from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_short_text(text):
    if '。' in text:
        texts = text.split('。')
        text_ = [tex for tex in texts if tex.rstrip() != '']
        return text_
    else:
        return []



def data_process():
    data_path = '../data/文本类语料/*.json'
    files = glob.glob(data_path)
    result = []
    templates = '对话上文:{0} 回复:{1}'
    for file_path in files:
        with open(file_path, 'r', encoding='utf8') as data:
            datas = json.load(data)
            train_data = datas['train']
            for data_ in train_data:
                try:
                    if len(data_) > 2:
                        print(data_)
                    else:
                        text1, text2 = data_[0].strip().replace('\r\n', ''), data_[1].strip().replace('\r\n', '')
                        texts = []
                        texts.extend(get_short_text(text1))
                        texts.extend(get_short_text(text2))

                        if texts:
                            for i in range(0, len(texts)-2, 2):
                                text_ = templates.format(texts[i], texts[i+1])
                                result.append(text_+'\n')
                except IndexError as e:
                    print(data_)

    with open(f'../data/文本类语料/train_{len(result)}.txt', 'w', encoding='utf8') as f:
        f.writelines(result)


def exract():
    dialog_re = "[“|\"]([\u4E00-\u9FA5A-Za-z0-9_\u3002\uff1b\uff0c\uff1a\uff08\uff09\u3001\uff1f\u300a\u300b\u2026]+)[”|\"]"
    dialog_re_compile = re.compile(dialog_re)
    result = []
    input_data_dir = '../data/essay/萨特传.txt'
    templates = '对话上文:{0} 回复:{1}'
    result_ = []
    headers = {
        'User-Agent': 'Apipost client Runtime/+https://www.apipost.cn/',
        'Content-Type': 'application/json',
    }

    with open(input_data_dir, 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            if find_text := dialog_re_compile.findall(line.strip()):
                result.extend(find_text)

        for i in range(0, len(result)-2, 2):
            data = {"texts": [result[i], result[i + 1]]}
            response = requests.post('http://36.189.234.222:60033/z/ranking/score', headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                score = response.json()['result'][0]['score']
                print('score: %0.4f' % score)
                if 0.99999 > score > 0.4:
                    text_ = templates.format(result[i], result[i + 1])
                    result_.append(text_ + '\n')
                else:
                    print((result[i], result[i + 1]))

    with open(f'../data/essay/萨特传_{len(result_)}.txt', 'w', encoding='utf8') as f:
        f.writelines(result_)


def split_data_txt():
    data_path = '../data/intent/意图数据-trigger.txt'

    with open(data_path, 'r', encoding='utf8') as f:
        texts = f.readlines()
        train_data, eval_datas = train_test_split(texts, test_size=0.2,shuffle=True)
        eval_data, test_data = train_test_split(eval_datas, test_size=0.2, shuffle=True)

        with open(f'../data/intent/train.txt', 'w', encoding='utf8') as f:
            f.writelines(train_data)
        with open(f'../data/intent/valid.txt', 'w', encoding='utf8') as f:
            f.writelines(eval_data)

        with open(f'../data/intent/test.txt', 'w', encoding='utf8') as f:
            f.writelines(test_data)


def split_data_csv(data_path):
    frame = pd.read_csv(data_path)


    train_data, eval_datas = train_test_split(frame[['question', 'answer']], test_size=0.2,shuffle=True)
    eval_data, test_data = train_test_split(eval_datas, test_size=0.2, shuffle=True)
    train_data.to_csv('../data/qa_match_cn/train.csv', index=False)
    eval_data.to_csv('../data/qa_match_cn/valid.csv', index=False)
    test_data.to_csv('../data/qa_match_cn/test.csv', index=False)


def enli_process():
    """
    处理恩利采访
    :return:
    """
    data_path = '../data/essay/恩利采访整理.txt'
    templates = '对话上文:{0} 回复:{1}'
    result_ = []
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f:
            if '.' in line:
                line = line.split('.')[1]
            lines = line.split('。')
            for i in range(0,len(lines)-1,2):
                try:
                    text = templates.format(lines[i], lines[i+1])
                    result_.append(text+'\n')
                except Exception as e:
                    print(lines)
    with open('../data/essay/恩利.txt', 'w', encoding='utf8') as f:
        f.writelines(result_)


def talk_process():
    """
    处理谈话录
    :return:
    """
    import fitz
    with fitz.open('../data/essay/卡夫卡第5卷随笔·谈话录.pdf') as f:
        text = ""
        print("pages:{0}".format(f.page_count))
        pages = f.page_count
        meta = f.metadata
        for page in f:

            text += page.get_text()

    print(text)


def process_csv(field_name1, field_name2):

    data = pd.read_csv('../data/intent/意图数据-trigger.csv')
    templates = '这句话表达了什么意图：{0} 意图：<{1}>'
    result_ = []
    for q, a in zip(data[field_name1], data[field_name2]):
        text = templates.format(q,a)
        result_.append(text+'\n')

    with open('../data/intent/意图数据-trigger.txt', 'a') as f:
        f.writelines(result_)


def process_csv_with_emotion(field_name1, field_name2, emotion_tag):

    emotion_dict = {'sad': '伤心', 'happy': '高兴', 'normal': '平静', 'angry': '愤怒'}
    data = pd.read_csv('../data/卡夫卡/theirs_all.csv')
    templates = '对话上文:{0} 回复:{1}的说：{2}'
    result_ = []
    for q, a, emo in zip(data[field_name1], data[field_name2], data[emotion_tag]):
        emo_ = emotion_dict[emo.lower()]
        text = templates.format(q, emo_, a)
        result_.append(text+'\n')

    with open('../data/卡夫卡/emotion_all.txt', 'a') as f:
        f.writelines(result_)

def process_qa():
    theirs_data_path = '../data/qa_match_cn/0526_theirs_base.csv'
    dialog_data_path = '../data/qa_match_cn/dialog_zh.csv'

    frame = pd.read_csv(theirs_data_path)
    frams2 = pd.read_csv(dialog_data_path)
    ta = pd.DataFrame()
    ta['question'] = frams2['question'].append(frame['Q'])
    ta['answer'] = frams2['answer'].append(frame['A'])

    ta.to_csv('../data/qa_match_cn/qa_match_cn.csv', index=False)

def process_test():
    data_path = '../data/qa_match_cn/test.csv'
    frame = pd.read_csv(data_path)
    ta = pd.DataFrame()
    ta['question'] = frame['question'].append(frame['answer'])
    ta['answer'] = frame['answer'].append(frame['question'])
    data = shuffle(ta[['question', 'answer']])
    data.to_csv('../data/qa_match_cn/qa_match_test.csv', index=False)


def test_intent():
    import requests
    import json

    url = "http://39.99.138.47:8015/z"

    payload = {
        "prompt": "",
        "number": 1,
        "length": 150,
        "top_p": 1,
        "temperature": 0.8,
        "strategy": "append"
    }
    headers = {
        'User-Agent': 'apifox/1.0.0 (https://www.apifox.cn)',
        'Content-Type': 'application/json'
    }
    result = []
    with open('../data/intent/test.txt', 'r') as f:
        for line in tqdm(f):
            line_ = line.strip('\n').split(' 意图：')[0] + ' 意图：'
            payload['prompt'] = line_

            response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
            text = response.json()['new_sentence'].replace('《', '').replace('》', '')
            line = line + ',' + text
            result.append(line + '\n')
            time.sleep(1)
    with open('../data/intent/test_result.txt', 'w') as f:
        f.writelines(result)


pattern = '^(回复|irs|伤心说|平静说|愤怒说|高兴说|伤心的说|平静的说|愤怒的说|高兴的说)'
complie_ = re.compile(pattern)

def process_colon(sentence, t=':', drop_list=None):
    """
    回复：开心的说：你好啊 -> 你好啊
    """
    if drop_list is None:
        drop_list = []
    sentence = sentence.strip()
    split_ = sentence.split(t)
    result = []
    contain_drop = False  # 是否存在drop_list
    for i in split_:
        if i in drop_list:
            contain_drop = True
            continue
        else:
            result.append(i)
    # 如果存在需要丢弃之前的
    if contain_drop:
        ans = t.join(result)

    else:
        ans = ''.join(result)

    while result := complie_.match(ans):
        _, end = result.span()
        ans = ans[end:]

    return ans

if __name__ == '__main__':
    # data_process()
    # exract()
    # erine()
    # split_data()
    # enli_process()
    # talk_process()
    # process_csv('多行文本', '中文意图')
    # process_qa()
    # split_data_csv('../data/qa_match_cn/qa_match_cn.csv')
    # split_data_txt()
    # process_test()
    # process_csv_with_emotion('Question', 'Answer', '情绪')
    ans = "  回复:高兴的说:这个我知道～“领航员”空间站的温度设定在华氏100度～"
    ans = process_colon(sentence=ans, t=':',drop_list=['irs', '伤心说', '平静说', '回复', '愤怒说', '高兴说', '伤心的说', '平静的说', '愤怒的说', '高兴的说'])
    print(ans)
    # test_intent()