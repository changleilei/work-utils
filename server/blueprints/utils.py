# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/1/21 6:23 下午
==================================="""
import requests

bad_sentence_url = "http://09e1fa9761124a3abc7e3b02c3c957f6-cn-zhangjiakou.alicloudapi.com/rct_tcl/toxic_cls_24L_en?text="


def is_toxic(text):
    response = requests.get(bad_sentence_url+text)
    if response.status_code == 200:
        response_json = response.json()
        if response_json.get("answer") == "2":
            return True
        else:
            return False
    else:
        return False


def pre_process(text_list):
    return text_list[:2]


def process_prompt(prompt, dialog_list, new_texts):
    """
    处理组合prompt
    :param prompt:
    :param dialog_list:
    :param new_texts:
    :return:
    """
    last_dialog_history = dialog_list[-1:]
    new_texts[0] = last_dialog_history[0]+new_texts[0]
    dialog_list = dialog_list[:-1]
    dialog_list.extend(new_texts)
    dialog_list.extend(last_dialog_history)
    prompt_ = prompt + "\n" + "\n".join(dialog_list)
    return prompt_.strip(), dialog_list
