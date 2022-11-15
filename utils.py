# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/11/15 6:20 下午
==================================="""
import re


def judge_zh(sentence):
    # 判断中文
    result = ''.join(re.findall(re.compile("([\u4E00-\u9FA5]+)"), sentence))
    if len(result) > len(sentence) / 2:
        return True
    else:
        return False

def judge_en(sentence):
    # 判断英文
    ignores = []
    result = ''.join(re.findall(r'[A-Za-z]', sentence))
    result = result.lower()
    for ignore in ignores:
        if ignore in result:
            result = result.replace(ignore, "")
    if len(result) > len(sentence) / 2:
        return True
    else:
        return False