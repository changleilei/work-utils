# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/1/12 11:23 上午
==================================="""
import os
import json
data_dir = "../data/douban"


def is_Chinese(ch):
    for c in ch:
        if not '\u4e00' <= c <= '\u9fff':
                return False
    return True


def douban():

    paths = os.listdir(data_dir)
    result = []
    for pa in paths:
        with open(data_dir+os.sep+pa, 'r', encoding='utf8') as f:
            for line in f.readlines():
                segment = line.split("\t")
                label = segment[0]
                if label == '1':
                    result_ = {'content':[]}
                    for sentence in segment[1:]:
                        # 英文使用空格隔开
                        se = []
                        for words in sentence.split(' '):
                            words = words.strip()
                            if is_Chinese(words):
                                se.append(words)
                            else:
                                se.append(" "+words)
                        se_ = "".join(se)
                        result_['content'].append(se_)
                    result.append(result_.__str__()+'\n')

    with open(data_dir+os.sep+'douban.json', 'w', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
    douban()