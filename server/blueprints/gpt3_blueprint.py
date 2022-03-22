# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/1/20 4:05 下午
==================================="""
import json

from flask import Blueprint, request, jsonify
import requests

from server.blueprints.utils import pre_process, process_prompt

generate_blueprints = Blueprint("genera", __name__)
gpt3_url = 'http://52.53.227.127:8000/completion'

gptj_url = 'http://39.103.143.138:8010/z'


@generate_blueprints.route("/gpt3", methods=["POST"])
def some_example():
    headers = {
        'Content-Type': 'application/json',
    }
    params = request.get_json()
    max_tokens = params.get('max_tokens', 100)
    temperature = params.get('temperature', 1)
    top_p = params.get('top_p', 1)
    n = params.get('n', 1)
    stop_words = params.get('stop_words', "\n")
    prompt = params.get('prompt')
    dialog = params.get('dialog')
    is_gpt3 = params.get('gpt3', False)
    epoch = params.get('epoch', 3)
    engine = params.get('engine', "davinci")
    if prompt is None:
        return jsonify("prompt param is required!"), 400

    if is_gpt3:
        url = gpt3_url
    else:
        url = gptj_url

    dialog_list = dialog.split("\n")
    prompt_ = prompt+"\n" + "\n".join(dialog_list)

    payload = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "n": n,
               "stream": False,  'prompt': prompt_, "engine": engine}
    count = 0
    with open("./generate.txt", 'w', encoding="utf8") as f:
        for j in range(10):

            for i in range(epoch):
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                if response.status_code == 200:
                    response_json = response.json()
                    if is_gpt3:
                        text = response_json['choices'][0]["text"]

                    else:
                        text = response_json['result'][0]

                    text_list = text.split("\n")
                    # 拿到前2轮或者前1轮
                    texts = pre_process(text_list)
                    if i == 0:
                        prompt_temp, dialog_list_temp = process_prompt(prompt, dialog_list, texts)
                    else:
                        prompt_temp, dialog_list_temp = process_prompt(prompt, dialog_list_temp, texts)
                    payload['prompt'] = prompt_temp
                else:
                    count += 1
                    continue
            if count == epoch:
                return jsonify({"result": "No generate result, please confirm generate server is alive!"})
            f.write(f"generate {j}\n")
            f.write(prompt_temp)

        return jsonify({"result": f"generate done:{j}"})








if __name__ == '__main__':
    url = 'http://09e1fa9761124a3abc7e3b02c3c957f6-cn-zhangjiakou.alicloudapi.com/rct_tcl/toxic_cls_24L_en?text=i wish i am not a girl'
    result_ = requests.get(url).json()
    print(result_)