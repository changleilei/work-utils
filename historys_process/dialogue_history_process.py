# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/9/30 4:37 下午
==================================="""
import json
from argparse import ArgumentParser
from json import JSONDecodeError

import pandas as pd
import requests
from tqdm import tqdm


def query_intent(text):
    intent_url = 'http://39.101.149.45:8015/z'
    payload = json.dumps({
        "prompt": f"句子：{text} 意图：",
        "number": 1,
        "length": 150,
        "top_p": 1,
        "top_k": 1,
        "temperature": 0.8,
        "strategy": "append"
    })
    headers = {
        'User-Agent': 'apifox/1.0.0 (https://www.apifox.cn)',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", intent_url, headers=headers, data=payload)

    category = response.json()["new_sentence"].replace("《", "").replace("》", "")
    return category

def process(data_path, save_path):
    datas = pd.read_csv(data_path)
    intents = json.load(open('intent.json', 'r'))
    result = {}
    for user, sentence, detail in zip(datas['用户/node'], datas['选择的回复'], datas['模型识别内容']):
        if user == 'user':
            if sentence in result:
                continue
            else:
                # try:
                #     d = json.loads(detail)
                #     intent = d['模型识别（意图，话题，user_profile等）']['intent']
                # except JSONDecodeError as e:
                #     continue

                # else:
                intent = query_intent(sentence)
                result[sentence] = intent
    final_result = {'sentence': [], 'intent': []}
    for key, intent in tqdm(result.items()):
        final_result['sentence'].append(key)

        if intent in intents:
            final_result['intent'].append(intent)
        else:
            final_result['intent'].append('intent')

    pd.DataFrame(final_result).to_csv(save_path, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/historys/history_qianduoduo.csv')
    parser.add_argument('--save_path',type=str, default='../data/historys/qianduoduo.csv')
    args = parser.parse_args()

    process(args.data_path, args.save_path)