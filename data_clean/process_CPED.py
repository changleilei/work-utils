# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/7/8 10:46 上午
==================================="""
import os
from collections import Counter
import pandas as pd
import glob
from tqdm import tqdm
emotion_labels = ['neutral', 'worried', 'grateful', 'negative-other', 'anger', 'disgust', 'sadness', 'astonished',
                  'happy', 'depress', 'positive-other', 'fear', 'relaxed']

less_emotion_labels = ['neutral', 'anger', 'happy', 'sadness']
count = []

less_emotion_labels2id = {key: value for value, key in enumerate(less_emotion_labels)}


def text_transform():
    data_dic_path = '../data/CPED'
    files = glob.glob(data_dic_path+os.sep+ '*.csv')

    for fi in files:
        frame = pd.read_csv(fi)
        d = {'label': [], 'text_a': []}
        txt_name = fi.split('_')[0].split('/')[-1] + '.tsv'

        for label, text in tqdm(zip(frame['Emotion'], frame['Utterance'])):
            if label in less_emotion_labels:
                d['label'].append(less_emotion_labels2id[label])
                d['text_a'].append(text.replace('\n', ','))

                count.append(label)
        f = pd.DataFrame(d)
        f.to_csv(data_dic_path+os.sep+txt_name, sep='\t', index=False)
        print(f'{txt_name} count : ')
        print(Counter(count))


def get_labels():
    data_path = '../data/CPED/test_split.csv'

    frame = pd.read_csv(data_path)
    labels = set()

    for s in frame['Emotion']:
        labels.add(s)
        if len(labels) == 13:
            break

    print(list(labels))

def process_colon(sentence, t=':', drop_list=None):
    """
    回复：开心的说：你好啊 -> 你好啊
    """
    if drop_list is None:
        drop_list = []
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
        return ans.strip(t)
    else:
        return ''.join(result)

if __name__ == '__main__':
    s = 'irs:回复:如下:你好啊'
    drop = ['回复', '开心的说']
    print(process_colon(s, t=':', drop_list=drop))