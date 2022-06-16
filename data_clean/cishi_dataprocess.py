# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/5/11 5:29 下午
==================================="""
import glob

import json
import re

import pandas as pd
import requests
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



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
    data_path = '../data/卡夫卡/base.txt'

    with open(data_path, 'r', encoding='utf8') as f:
        texts = f.readlines()
        train_data, eval_datas = train_test_split(texts, test_size=0.2,shuffle=True)
        eval_data, test_data = train_test_split(eval_datas, test_size=0.2, shuffle=True)

        with open(f'../data/卡夫卡/0527/train.txt', 'w', encoding='utf8') as f:
            f.writelines(train_data)
        with open(f'../data/卡夫卡/0527/valid.txt', 'w', encoding='utf8') as f:
            f.writelines(eval_data)

        with open(f'../data/卡夫卡/0527/test.txt', 'w', encoding='utf8') as f:
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

    data = pd.read_csv('../data/卡夫卡/0527_base.csv')
    templates = '对话上文:{0} 回复:{1}'
    result_ = []
    for q, a in zip(data[field_name1], data[field_name2]):
        text = templates.format(q,a)
        result_.append(text+'\n')

    with open('../data/卡夫卡/base.txt', 'a') as f:
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

if __name__ == '__main__':
    # data_process()
    # exract()
    # erine()
    # split_data()
    # enli_process()
    # talk_process()
    # process_csv('Q', 'A')
    # process_qa()
    # split_data_csv('../data/qa_match_cn/qa_match_cn.csv')
    # split_data_txt()
    # process_test()
    process_csv_with_emotion('Question', 'Answer', '情绪')