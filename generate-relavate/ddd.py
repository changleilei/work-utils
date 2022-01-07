# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2021/12/23 6:31 下午
==================================="""
import json
import pandas as pd
data_dir = "../data/emotion9.1w.csv"


def func():
    datas = pd.read_csv(data_dir)
    dict_ = {"text": [], "label": []}
    for text, label in zip(datas['text'], datas['rct-label']):
        dict_['text'].append(text)
        if label > 1:

            dict_['label'].append(label-1)
        else:
            dict_['label'].append(label)

    frame = pd.DataFrame(dict_)
    frame.to_csv('emotion_5_label.csv')
    print(frame.groupby(frame['label']).count())


if __name__ == '__main__':
    func()