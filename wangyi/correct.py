# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/3/9 4:19 PM
==================================="""
# -*- coding: utf-8 -*-
import sys
import uuid
import requests
import hashlib
import time
import time

# sys.setdefaultencoding('utf-8')

YOUDAO_URL = 'https://openapi.youdao.com/correct_writing_cn_text'
APP_KEY = '1bc8cc51c56cd938'
APP_SECRET = 'OXFmFHPjjxVADEk8LajzvOYLaZiMIBAB'


def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def do_request(data):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    return requests.post(YOUDAO_URL, data=data, headers=headers)


def connect():
    q = "春天来了，大地万物开始复苏。树木抽出嫩绿的新芽，花儿绽放出五彩斑斓的颜色，草地上也长出了茂密的青草。人们开始换上轻便的衣服，感受温暖的阳光和清新的空气，享受着春天带来的愉悦和生机。同时，春天也是一个适宜的季节，人们可以在这个季节里多出门散步、运动、旅行，增强身体的健康和抵抗力。春天是美好的，也是充满希望和活力的季节。\n在春天，自然界的各种生命都充满了活力和生机。不仅是植物和动物，春天也是人类的季节。人们在春季中不断追求着自己的梦想和目标，努力地工作和学习，为自己的未来打下坚实的基础。同时，春天也是一个充满爱和希望的季节，许多人在这个季节里相识、相恋，共同经历着生命中的美好时光。春天是一个充满着生命和爱的季节，也是我们回归大自然、感受生命的绝佳时机。"
    grade = "g1"
    data = {}
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    data['signType'] = "v3"
    data['grade'] = grade
    response = do_request(data)
    contentType = response.headers['Content-Type']
    if contentType == "audio/mp3":
        millis = int(round(time.time() * 1000))
        filePath = "合成的音频存储路径" + str(millis) + ".mp3"
        fo = open(filePath, 'wb')
        fo.write(response.content)
        fo.close()
    else:
        print(response.text)


if __name__ == '__main__':
    connect()