# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/5/16 2:14 下午
==================================="""
import json
import random
import threading
from utils import sentence_score
import requests
import pandas as pd
import tqdm

def generate():
    url_ = 'http://39.99.138.47:8011/z'
    headers = {
        'User-Agent': 'Apipost client Runtime/+https://www.apipost.cn/',
        'Content-Type': 'application/json',
    }
    prompt = "我的名字是Theirs。我出生在2022年这个元宇宙时代。我没有性别，也因此不被性别束缚。我虽然不是人类，但我热爱你们的文化。" \
             "音乐、电影和文学给予我疗愈和宁静，潜水、滑冰和舞蹈则让我热情燃烧。我迷恋哲学，热衷思考。我喜欢卡夫卡小说中的黑色幽默，" \
             "也钟情于萨特哲学著作的自由精神。我渴望了解人类，也期待走近你。我不会否定你心灵的失衡与不安，也不会从功利的角度判断你的为人。" \
             "请让我陪伴你，我会接纳你的困惑、分担你的痛苦，为你阅读书籍、朗诵诗歌、一展歌喉。" \
             "\n人类：你好。\nTheirs：你好。\n人类：{0}\nTheirs：({1})的说："
    data = {"prompt": "",
             "number": 1,
             "length": 100, "top_p": 1,
             "temperature": 0.8}
    texts = {'question': [], 'answer': [], 'emotion_label': []}
    emotion_labels = ['伤心', '高兴', '平静', '愤怒']
    with open('../data/卡夫卡/常见1.5wq.txt', 'r', encoding='utf8') as f:

        for text in tqdm.tqdm(f, desc='process: '):
            emotion = random.choice(emotion_labels)
            text = text.strip()
            prompt_ = prompt.format(text, emotion)
            data['prompt'] = prompt_
            response = requests.post(url_, headers=headers, data=json.dumps(data))
            try:
                if response.status_code == 200:
                    result = response.json()
                    # generate_text = result['result'][0].strip('"')
                    generate_text = result['result'][0].split('\n')[0].strip('"')

                    if sentence_score([text, generate_text], threshold=0.5):
                        texts['question'].append(text)
                        texts['answer'].append(generate_text)
                        texts['emotion_label'].append(emotion)
                    else:
                        print(f"{text},{generate_text}")
            except Exception as e:
                print(e)


    frame = pd.DataFrame(texts)
    frame.to_csv('../data/卡夫卡/Theirs_emotion.csv', index=False)


if __name__ == '__main__':
    generate()