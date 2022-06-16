# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/6/7 3:28 下午
==================================="""


import glob
import os

from sklearn.model_selection import train_test_split

base_dir = '../data/chn/senteval_cn/'


def get_files(base_dir_):
    dirs = [os.path.join(base_dir_, name) for name in os.listdir(base_dir_)
             if os.path.isdir(os.path.join(base_dir_, name))]
    files = [os.path.join(d, name) for d in dirs for name in os.listdir(d) if name.endswith('.data')]

    return files

def process_datas(files):
    result = []
    for fi in files:
        with open(fi, 'r', encoding='utf8') as f:
            if 'STS-B' in fi:
                for t in f.readlines():
                    texts = t.split('\t')
                    score = float(texts[2]) / 5.0
                    texts[2] = str(score)
                    tt = '\t'.join(texts)
                    result.append(tt+'\n')
            else:
                result.extend(f.readlines())

    with open('../data/chn/all.txt', 'w', encoding='utf8') as f:
        f.writelines(result)


def split_data_csv(data_path):
    with open(data_path, 'r') as f:
        datas = f.readlines()
    train_data, eval_datas = train_test_split(datas, test_size=0.2,shuffle=True)
    eval_data, test_data = train_test_split(eval_datas, test_size=0.2, shuffle=True)

    write_txt('../data/chn/train_datas/train.txt', train_data)
    write_txt('../data/chn/train_datas/valid.txt', eval_data)
    write_txt('../data/chn/train_datas/test.txt', test_data)


def write_txt(data_path, datas):
    with open(data_path, 'w', encoding='utf8') as f:
        f.writelines(datas)


if __name__ == '__main__':
    # files = get_files(base_dir)
    # process_datas(files)
    split_data_csv('../data/chn/all.txt')