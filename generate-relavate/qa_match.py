# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/3/11 10:09 上午
==================================="""
import requests
import json
headers = {
    'User-Agent': 'Apipost client Runtime/+https://www.apipost.cn/',
    'Content-Type': 'application/json',
}

# data = {"context":[ "Hi boy ,how is it going?", "Who are you?", ], "response":"I am your boss." }  # 0.77
data = {"context":["Who are you?" ], "response": "I'm your sister"}

response = requests.post('http://39.99.249.45:8060/z', headers=headers, data=json.dumps(data))
print(response.json())