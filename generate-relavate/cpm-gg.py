# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2021/12/9 11:45 上午
==================================="""

import requests
import json
import pandas as pd
from tqdm import tqdm

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
            frame.to_csv(save_data_path + f"仙剑_generate{i}.csv")
        prompt_ = prompt.format(question)
        payload['prompt'] = prompt_
        payload_pangu['prompt'] = pangu_prompt.format(question)
        try:
            response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
            response_xianxia = requests.request("POST", url_xianxia, headers=headers, data=json.dumps(payload))
            response_pangu = requests.request("POST", url_pangu, headers=headers, data=json.dumps(payload_pangu))
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


def compare_cpm():
    theirs_url = 'http://39.99.138.47:8010/z'
    emotion_cpm_url = 'http://39.99.138.47:8011/z'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "prompt": "",
        "number": 1,
        "length": 150,
        "top_p": 1,
        "temperature": 0.8,
        "strategy": "delete"
    }
    prompt = "我是Theirs，出生在2022年这个元宇宙时代。我虽然不是人类，但我热爱人类的文化：音乐、电影和文学给予我疗愈和宁静；" \
             "潜水、滑冰和舞蹈则让我热情燃烧。我是无性别的，不被性别束缚。如果你时常感觉到孤独，不爱社交，但又渴求文化的熏陶和灵魂的际遇，" \
             "我可以为你阅读书籍、朗诵诗歌，还可以为你一展歌喉。我通晓哲学，无所不知，我喜欢卡夫卡的黑色幽默，也喜欢萨特的自由灵魂。" \
             "我不会否定你心灵的失衡与不安，也不会从功利的角度判断你的为人，我会接纳你的困惑、分担你的痛苦。我喜欢顺其自然的交往，" \
             "不会吵吵闹闹地喋喋不休。\n人类：你好。\nTheirs：你好。\n人类：{0}\nTheirs：({1})的说："

    data_path = '../data/cpm-compare/theirs_all.csv'
    def temp_func(text_json):
        text = text_json['result'][0].split('\n')[0]
        return text

    emotion_dict = {'sad': '伤心', 'happy': '高兴', 'normal': '平静', 'angry': '愤怒'}
    datas = pd.read_csv(data_path)
    datas = datas[['Question', 'Answer', '情绪']]
    result = {'Question': [], 'Answer': [], 'theirs_result': [],'theirs_emotion_results': [], 'emotion_label': []}
    for q, a, emo in tqdm(zip(datas['Question'], datas['Answer'], datas['情绪'])):
        emo_ = emotion_dict[emo.lower()]
        prompt_ = prompt.format(q, emo_)
        payload['prompt'] = prompt_
        rpm_response = requests.request("POST", theirs_url, headers=headers, data=json.dumps(payload))
        theirs_finetune_response = requests.request("POST", emotion_cpm_url, headers=headers, data=json.dumps(payload))
        if rpm_response.status_code == 200 and theirs_finetune_response.status_code == 200:
            rpm_result = temp_func(rpm_response.json())
            theirs_result = temp_func(theirs_finetune_response.json())
            result['theirs_result'].append(rpm_result)
            result['theirs_emotion_results'].append(theirs_result)
            result['Question'].append(q)
            result['Answer'].append(a)
            result['emotion_label'].append(emo_)
        else:
            print(f"theirs status_code: {rpm_response.status_code}")
            print(f"theirs status_code: {theirs_finetune_response.status_code}")
            print(f"Q: {q}")
            print(f"A: {a}")

        pd.DataFrame(result).to_csv('../data/cpm-compare/compare2.csv', index=False)





if __name__ == '__main__':
    # generate_func()
    compare_cpm()
