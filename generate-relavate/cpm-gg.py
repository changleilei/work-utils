# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2021/12/9 11:45 上午
==================================="""

import requests
import json
import pandas as pd

url_xianxia = "http://models-repo.rctdev.cn:8030/z"  # CPM 仙侠
url = "http://models-repo.rctdev.cn:8010/z"  # CPM
url_pangu = "http://39.103.143.138:8082/z"
data_path = "../data/仙剑_chitchat.csv"
save_data_path = "../data/"


def generate_func():
    payload = {
        "prompt": '',
        "number": 1,
        "length": 100,
        "top_p": 0.497,
        "temperature": 0.516
    }
    payload_pangu = {
    "number": 1,
    "length": 1000,
    "max_num": 200,
    "prompt": "",
    "top_p": 0.9,
    "top_k": 100,
    "temperature": 1
    }
    headers = {
        'Content-Type': 'application/json'
    }
    result = {'question': [], 'cpm': [], 'cpmx_xianxia': [], 'pangu': []}
    prompt = "李逍遥原是乡下客栈的店小二，与照顾自己长大的婶婶相依为命。他天资聪颖，因一壶酒被酒剑仙传授了蜀山仙剑派剑术，学成之后，" \
             "为求药前去仙灵岛，在岛上与赵灵儿一见钟情，并且成亲，出岛后因拜月教的秘药而失忆。重新认识赵灵儿后,帮助灵儿回到南诏，" \
             "后结识林月如，阿奴等人，历经重重磨难，成长为一代旷世奇侠。人类:你好呀。\n李逍遥:你好。\n人类:{0}\n李逍遥:"
    pangu_prompt = "问：{}答："
    datas = pd.read_csv(data_path)

    def temp_func(text_json):
        text = text_json['result'][0].split('\n')[0]
        return text

    for i, question in enumerate(datas['question']):
        if i % 500 == 0 and i != 0:
            print(f"already generate {i} items")
            frame = pd.DataFrame(result)
            frame.to_csv(save_data_path+f"仙剑_generate{i}.csv")
        prompt_ = prompt.format(question)
        payload['prompt'] = prompt_
        payload_pangu['prompt'] = pangu_prompt.format(question)
        try:
            response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
            response_xianxia = requests.request("POST", url_xianxia, headers=headers, data=json.dumps(payload))
            response_pangu = requests.request("POST", url_pangu, headers=headers,data=json.dumps(payload_pangu))
            cpm_result = temp_func(response.json())
            cpmx_xianxia_result = temp_func(response_xianxia.json())
            pangu_result = temp_func(response_pangu.json())
        except Exception as e:
            print(f"error was happen {e}")
            continue

        result['question'].append(question)
        result['cpm'].append(cpm_result)
        result['cpmx_xianxia'].append(cpmx_xianxia_result)
        result['pangu'].append(pangu_result)


if __name__ == '__main__':
    generate_func()
