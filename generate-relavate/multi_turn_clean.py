# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2021/12/13 5:49 下午
==================================="""
import json



def func():
    data_dir = "../data/vocab.json"
    datas = json.load(open(data_dir, 'r'), encoding='utf8')

    with open("../data/multi_turn/vocab.json", 'w', encoding='utf8') as f:
        json.dump(datas, f, ensure_ascii=False)


if __name__ == '__main__':
    func()