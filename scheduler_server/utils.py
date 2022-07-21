# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/2/25 5:35 下午
==================================="""
from typing import Union, List

from emotion_detect import get_emotion_label

def result_format(result):
    return {"result": result}


def process_history(historys, emotion_historys, user_name, bot_name):
    """将history 和 emotion_history 拼接成GPT输入"""
    utterance = ''

    if emotion_historys:
        for history_round, emotion_round in zip(historys, emotion_historys):

            user_utterance = history_round[0].get(user_name)

            bot_utterance = history_round[1].get(bot_name)

            user_emotion = emotion_round[0].get('user_emotion', None)
            bot_emotion = emotion_round[1].get('bot_emotion', None)
            # user_emotion is not None
            if user_emotion is not None:
                utterance += f"{user_name}: ({user_emotion}){user_utterance}\n"
            else:
                utterance += f"{user_name}: {user_utterance}\n"

            # bot_emotion is not None
            if bot_emotion is not None:
                utterance += f"{bot_name}: ({bot_emotion}){bot_utterance}\n"
            else:
                utterance += f"{bot_name}: {bot_utterance}\n"
    else:
        for history_round in historys:
            user_utterance = history_round[0].get(user_name)

            bot_utterance = history_round[1].get(bot_name)
            utterance += f"{user_name}: {user_utterance}\n{bot_name}: {bot_utterance}\n"

    return utterance


def pre_process(text_list):
    return text_list[:2]


def process_prompt(prompt, dialog_his, new_texts, bot_name):
    """
    处理组合prompt
    :param bot_name: bot_name for remove
    :param prompt:
    :param dialog_list:
    :param new_texts:
    :return:
    """
    texts = []
    for text in new_texts:
        text = text.strip()
        bot_name = bot_name.strip()
        if text != bot_name:
            texts.append(text)
    new_text = '\n'.join(texts).strip()
    dialog_his = dialog_his + new_text
    prompt = prompt + dialog_his
    return prompt, dialog_his


def marking_sentence(text: Union[str, List]):
    """
    将送入的句子打标
    :param text:
    :return:
    """
    if not text:
        return None

    if isinstance(text, List):
        text_for_label = []
        names = []
        for t in text:
            if ':' in t:
                temp_text = t.split(':')[-1]
                name = t.split(':')[0]
                names.append(name)
                text_for_label.append(temp_text)
            else:
                names.append('')
                text_for_label.append(t)
        labels = get_emotion_label(text_for_label)
        label_result = []
        if labels:
            for i, label in enumerate(labels):
                if label:
                    temp_text = names[i].strip() + ": ({0})".format(label) + text_for_label[i].strip()
                else:
                    temp_text = names[i].strip() + ": " + text_for_label[i].strip()
                temp_text = temp_text.strip(':')
                label_result.append(temp_text)
            return label_result
        else:
            return text
    else:
        if ':' in text:
            label = get_emotion_label(text.split(':')[-1])
            if label:
                return text.split(':')[0].strip() + ': ({0})'.format(label) + text.split(':')[-1].strip()
            else:
                return text
