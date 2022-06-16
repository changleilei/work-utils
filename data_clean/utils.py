# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/5/16 2:14 下午
==================================="""
import json

import requests


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def all_is_Chinese(seq):
    for char in seq:
        cp = ord(char)
        if not is_chinese_char(cp):
            return False
    return True

def sentence_score(sentences, threshold=0.5):
    headers = {
        'User-Agent': 'Apipost client Runtime/+https://www.apipost.cn/',
        'Content-Type': 'application/json',
    }
    data = {"texts": sentences}
    response = requests.post('http://39.101.149.45:8074/ranking/cosin', headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        score = response.json()['result'][0]['score']
        # print('score: %0.4f' % score)
        if 0.99999 > score > threshold:
            return True
        else:
            return False

if __name__ == '__main__':
    t = '你好啊,'
    for s in t:
        c = ord(s)
        if is_chinese_char(c):
            print(True)
    print(t.replace(',', '，'))