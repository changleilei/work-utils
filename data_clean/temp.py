# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/9/13 9:38 下午
==================================="""

import json
from json import JSONDecodeError

import pandas as pd


def func():
    data_path = '../data/zhuti/all_theme_intent.txt'
    save_path = '../data/zhuti/all_theme_intent1.jsonl'
    with open(data_path, 'r') as f, open(save_path, 'w', encoding='utf8', newline='\n') as wf:
        for line in f:
            try:
                data = json.loads(line)
                wf.write(json.dumps({'text': data}, ensure_ascii=False)+'\n')
            except JSONDecodeError as e:
                print(data)


def func2():
    data_path = '../data/emotion/情绪.txt'
    save_path = '../data/emotion/情绪.csv'
    result = {'sentence': [], 'emotion': []}
    with open(data_path, 'r') as f, open(save_path, 'w', encoding='utf8') as wf:
        for line in f:
            try:
                splits = line.split('情绪是:')
                text, label = splits[0].replace('判断以下句子的情绪:', ''), splits[1].strip()
                text = text.strip('\n')
                result['sentence'].append(text)
                result['emotion'].append(label)
            except JSONDecodeError as e:
                print(e)

        pd.DataFrame(result).to_csv(save_path, index=False)


def func3():
    """测试summary的能力 """
    data_path = '../data/multi_turn/dialog_zh.json'
    save_path = '../data/temp/summary1.csv'

    import requests
    import json
    with open(data_path, 'r', encoding='utf8') as f, open(save_path, 'w', encoding='utf8') as w_f:
        datas = json.load(f)
        w_f.write('sentence,summary'+'\n')
        for i, da in enumerate(datas):
            if i > 50:
                w_f.close()
                break
            content = da['content']
            sentences_str = ''.join(content[:min(8, len(da['content']))])

            url = "http://192.168.0.181:8078/summarizer"

            payload = json.dumps({
                "sentence": [sentences_str]
            })
            headers = {
                'User-Agent': 'apifox/1.0.0 (https://www.apifox.cn)',
                'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)
            result = response.json()['result'][0]
            w_f.write(sentences_str+','+result+'\n')

def func4():
    data_path = '../data/intent/v2-v3.csv'
    save_path = '../data/intent/en_intent_map.json'
    data = pd.read_csv(data_path)
    result = {}
    for key, value in zip(data['v2'], data['v3']):
        result[key] = value
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False)

if __name__ == '__main__':
    func4()