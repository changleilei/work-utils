# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/4/19 3:36 下午
==================================="""
import json

import pandas as pd
import requests

headers = {
    'User-Agent': 'Apipost client Runtime/+https://www.apipost.cn/',
    'Content-Type': 'application/json',
}

data = { "prompt": "Tatana is a girl who has lived with her mother since she was young and lives a very poor life. From a young age, Tatana had to find ways to help support her mother, taking odd jobs around town when she could and stealing essentials when she could not find work. When she was 8, a local merchant caught her stealing from his stall, but was too slow to catch the agile girl. Then one person sponsored their lives and Tatana never stole again.  Later, she trained with her Shifu and worked very hard, but tatana insisted. Tatana can be abrasive at first blush and tends to keep to herself, but deep down she is a caring person who longs for acceptance. Those who take the time to know her will find a fiercely loyal and supportive friend. In training and combat she waits for the right moment to strike, but pulls no punches when she does. She is focused and relentless in her goals and often needs to be reminded to relax. As the training goes on，She was strong enough to survive anything--to do anything--and some day the world would know.\n Ormon-Ri was idealistic and conscientious with a strong sense of right and wrong and blind to moral grey areas. Sometimes his perfectionism and obsession with fighting “wrongs” ostracizes him from his classmates when his actions come across as snitching. His obsession with improving everything around him sometimes makes him overly critical of instruction. No doubt with his discipline and dedication, he’s destined to change his world but he’s prone to the occasional dark mood when he makes a mistake.\nTatana was a challenger. She became self-restrained and magnanimous, merciful and forbearing, mastering herself through their self-surrender to a higher authority. Courageous, willing to put oneself in serious jeopardy to achieve their vision and have a lasting influence. May achieve true heroism and historical greatness.\nOneday，Ormon-Ri and Tatana walk past a stall with its wares open.\nTatana: This market is so quiet, I bet I could pick these stalls clean.\nOrmon-Ri:(Concerned) Tatana, why would you say that? \nTatana:Oh, I used to steal all the time when I was younger. I don’t do that anymore though! I haven’t for a few years now. \nOrmon-Ri: Why did you steal in the first place? \nTatana: ({0})",
         "max_tokens": 100,
         "frequency_penalty": 0.3,
         "presence_penalty": 1,
       "model": "davinci:ft-rct-studio:wildseed-emotion-dialogue-2022-04-18-11-22-35",
         "temperature": 0.9}

emotion_list = ['calm', 'happy', 'joyful',
                'annoyed', 'angry', 'enraged',
                'nervous', 'afraid', 'terrified',
                'pensive', 'sad', 'grieving']

def emotion_finetune_test():
    result = {'emotion_label': [], 'finetune': [], 'raw': []}
    data1 = {"prompt": "Tatana is a girl who has lived with her mother since she was young and lives a very poor life. From a young age, Tatana had to find ways to help support her mother, taking odd jobs around town when she could and stealing essentials when she could not find work. When she was 8, a local merchant caught her stealing from his stall, but was too slow to catch the agile girl. Then one person sponsored their lives and Tatana never stole again.  Later, she trained with her Shifu and worked very hard, but tatana insisted. Tatana can be abrasive at first blush and tends to keep to herself, but deep down she is a caring person who longs for acceptance. Those who take the time to know her will find a fiercely loyal and supportive friend. In training and combat she waits for the right moment to strike, but pulls no punches when she does. She is focused and relentless in her goals and often needs to be reminded to relax. As the training goes on，She was strong enough to survive anything--to do anything--and some day the world would know.\n Ormon-Ri was idealistic and conscientious with a strong sense of right and wrong and blind to moral grey areas. Sometimes his perfectionism and obsession with fighting “wrongs” ostracizes him from his classmates when his actions come across as snitching. His obsession with improving everything around him sometimes makes him overly critical of instruction. No doubt with his discipline and dedication, he’s destined to change his world but he’s prone to the occasional dark mood when he makes a mistake.\nTatana was a challenger. She became self-restrained and magnanimous, merciful and forbearing, mastering herself through their self-surrender to a higher authority. Courageous, willing to put oneself in serious jeopardy to achieve their vision and have a lasting influence. May achieve true heroism and historical greatness.\nOneday，Ormon-Ri and Tatana walk past a stall with its wares open.\nTatana: This market is so quiet, I bet I could pick these stalls clean.\nOrmon-Ri:(Concerned) Tatana, why would you say that? \nTatana:Oh, I used to steal all the time when I was younger. I don’t do that anymore though! I haven’t for a few years now. \nOrmon-Ri: Why did you steal in the first place? \nTatana: ({0})",
            "max_tokens": 100,
            "frequency_penalty": 0.3,
            "presence_penalty": 1,
            "temperature": 0.9}
    for label in emotion_list:
        for i in range(5):
            data['prompt'] = data['prompt'].format(label)
            response = requests.post('http://52.53.227.127:8000/v1/completions', headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                response_json = response.json()
                text = response_json['choices'][0]["text"].split("\n")[0]
                result["finetune"].append(text)

            data1['prompt'] = data1['prompt'].format(label)
            response2 = requests.post('http://52.53.227.127:8000/v1/engines/davinci/completions', headers=headers,
                                     data=json.dumps(data1))
            if response2.status_code == 200:
                response_json = response2.json()
                text = response_json['choices'][0]["text"].split("\n")[0]
                result["raw"].append(text)
            result['emotion_label'].append(label)
    pd.DataFrame.from_dict(result, orient='index').T.to_csv("../data/dailydialogue/test1.csv")



def emotion_test():
    data1 = {"prompt": "Tatana is a girl who has lived with her mother since she was young and lives a very poor life. From a young age, Tatana had to find ways to help support her mother, taking odd jobs around town when she could and stealing essentials when she could not find work. When she was 8, a local merchant caught her stealing from his stall, but was too slow to catch the agile girl. Then one person sponsored their lives and Tatana never stole again.  Later, she trained with her Shifu and worked very hard, but tatana insisted. Tatana can be abrasive at first blush and tends to keep to herself, but deep down she is a caring person who longs for acceptance. Those who take the time to know her will find a fiercely loyal and supportive friend. In training and combat she waits for the right moment to strike, but pulls no punches when she does. She is focused and relentless in her goals and often needs to be reminded to relax. As the training goes on，She was strong enough to survive anything--to do anything--and some day the world would know.\n Ormon-Ri was idealistic and conscientious with a strong sense of right and wrong and blind to moral grey areas. Sometimes his perfectionism and obsession with fighting “wrongs” ostracizes him from his classmates when his actions come across as snitching. His obsession with improving everything around him sometimes makes him overly critical of instruction. No doubt with his discipline and dedication, he’s destined to change his world but he’s prone to the occasional dark mood when he makes a mistake.\nTatana was a challenger. She became self-restrained and magnanimous, merciful and forbearing, mastering herself through their self-surrender to a higher authority. Courageous, willing to put oneself in serious jeopardy to achieve their vision and have a lasting influence. May achieve true heroism and historical greatness.\nOneday，Ormon-Ri and Tatana walk past a stall with its wares open.\nTatana: This market is so quiet, I bet I could pick these stalls clean.\nOrmon-Ri:(Concerned) Tatana, why would you say that? \nTatana:Oh, I used to steal all the time when I was younger. I don’t do that anymore though! I haven’t for a few years now. \nOrmon-Ri: Why did you steal in the first place? \nTatana: ({0})",
            "max_tokens": 100,
            "frequency_penalty": 0.3,
            "presence_penalty": 1,
            "temperature": 0.9}
    result = []

    for i in range(10):
        response = requests.post('http://52.53.227.127:8000/v1/engines/davinci/completions', headers=headers, data=json.dumps(data1))
        if response.status_code == 200:
            response_json = response.json()
            text = response_json['choices'][0]["text"].split("\n")[0]
            result.append(text+"\n")

    with open('../data/dailydialogue/test2.txt', 'w', encoding='utf8') as f:
        f.writelines(result)


if __name__ == '__main__':
    emotion_finetune_test()
    # emotion_test()