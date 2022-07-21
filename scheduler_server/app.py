# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/7/21 11:29 上午
==================================="""
# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/1/25 11:34 上午
==================================="""
import json
import re
import time
from pytz import utc
import requests
from flask import Flask, request, jsonify
from utils import result_format, process_history, pre_process, process_prompt, marking_sentence
from flask_apscheduler.scheduler import APScheduler, BackgroundScheduler
from logger import get_logger
from flask_cors import CORS
import neuralcoref
import spacy

app = Flask(__name__)

request_times = 0

gpt3_url = ""
gpt3_dialog_url = ""
neuralcoref_url = ""

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

logger = get_logger()

headers = {
    'Content-Type': 'application/json',
}

@app.after_request
def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp


@app.route("/dialogue", methods=["POST"])
def dialogue():
    """
    This is a emotion dialogue API.

    """
    params = request.get_json()
    max_tokens = params.get('max_tokens', 100)
    temperature = params.get('temperature', 1)
    prompt = params.get('prompt')
    top_p = params.get('top_p', 1)
    n = params.get('n', 1)
    engine = params.get('engine', "davinci")
    if not prompt:
        return jsonify({"result": "prompt is require"})
    payload = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "n": n,
               "stream": False,  'prompt': prompt}

    logger.info(payload)
    response = requests.post(gpt3_url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        response_json = response.json()
        text = response_json['choices'][0]["text"].split("\n")[0]
        return jsonify(result_format({"sentence": text, "score": 0.0}))
    else:
        return jsonify(result_format({"sentence": "Ops! I have no idea about this.", "score": 0.0})), 500


@app.route("/emotiondialogue", methods=["POST"])
def emotion_dialogue():
    """
        This is a dialogue API.

    """
    params = request.get_json()
    max_tokens = params.get('max_tokens', 100)
    temperature = params.get('temperature', 1)
    prompt = params.get('prompt')
    top_p = params.get('top_p', 1)
    n = params.get('n', 1)
    input_text = params.get('user_question', '')
    historys = params.get('historys', [])  # [[dict, dict], [dict, dict]]
    emotion_historys = params.get('emotion_historys', [])
    node_current_emotion = params.get('node_current_emotion', '')
    user_current_emotion = params.get('user_current_emotion', '')
    user_name = params.get('user_name')
    node_name = params.get('node_name')

    if not input_text:
        return jsonify(result_format({"sentence": "Ops! I have no idea about this.", "score": 0.0})), 400

    if historys:
        # 有历史
        utterance = process_history(historys, emotion_historys, user_name, node_name)
        # 拼接当前对话状态和将要进行的对话
        if user_current_emotion:
            utterance += f'{user_name}: ({user_current_emotion}){input_text}\n{node_name}: '
        else:
            utterance += f'{user_name}: {input_text}\n{node_name}: '

        if node_current_emotion:
            utterance += f'({node_current_emotion})'
    else:
        # 无历史
        if user_current_emotion:
            utterance = f'{user_name}: ({user_current_emotion}){input_text}\n{node_name}: '
        else:
            utterance = f'{user_name}: {input_text}\n{node_name}: '

        if node_current_emotion:
            utterance += f'({node_current_emotion})'

    prompt = prompt + '\n' + utterance

    # 去除括号内的内容
    pattern = re.compile("[\(](.*?)[\)]")
    payload = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "n": n,
               "stream": False,  'prompt': prompt}

    logger.info(payload)
    response = requests.post(gpt3_url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        response_json = response.json()
        text = response_json['choices'][0]["text"].split("\n")[0]
        # 如果有括号替换为空
        text = pattern.sub('', text)
        return jsonify(result_format({"sentence": text, "score": 0.0}))
    else:
        return jsonify(result_format({"sentence": "Ops! I have no idea about this.", "score": 0.0})), 500


@app.route("/wildseed_dialog", methods=["POST"])
def wildseed_dialog():
    global request_times
    if request_times > 40:
        return jsonify({"result": "The amount of generation is limited daily!"}), 500
    not_talk = "This is a chatbot. It can't discuss racial discrimination, gender discrimination, politics and other related issues.\n"
    params = request.get_json()
    prompt = params.get('backstory')
    enneagram = params.get('enneagram')
    scene = params.get('scene')
    dialogue_history = params.get('context')
    turns = params.get('turns', 2)
    times = params.get('times', 1)
    bot_name = dialogue_history.split('\n')[-1]

    prompt = not_talk + prompt
    prompt_ = prompt+'\n'+enneagram+'\n'+scene+'\n'+dialogue_history

    payload = {"max_tokens": 200, "temperature": 0.7, "top_p": 1, "n": 1,
               "stream": False,  'prompt': prompt_}
    logger.info(f"Prompt: {prompt_}")
    result = []
    for j in range(times):
        count = 0
        dialog_history_temp = ''
        for i in range(turns):
            response = requests.post(gpt3_dialog_url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                response_json = response.json()
                # if is_gpt3:
                text = response_json['choices'][0]["text"]

                # else:
                #     text = response_json['result'][0]
                text = text.strip()
                text_list = text.split("\n")
                # 拿到前2轮或者前1轮
                texts = pre_process(text_list)

                if i == 0:
                    prompt_temp, dialog_history_temp = process_prompt(prompt, dialogue_history, texts, bot_name)

                else:
                    dialog_history_temp_ = dialogue_history if not dialog_history_temp else dialog_history_temp + '\n' + bot_name
                    prompt_temp, dialog_history_temp = process_prompt(prompt, dialog_history_temp_, texts, bot_name)
                # # 第二轮的时候不使用情绪
                # if ':' in bot_name:
                #     bot_name_temp = bot_name.split(':')[0]
                # else:
                #     bot_name_temp = bot_name
                payload['prompt'] = prompt_temp + '\n' + bot_name
                if i == turns-1:
                    result.append(dialog_history_temp)
            else:
                count += 1
                continue

        if count == times:
            return jsonify({"result": "No generate result, if many times, please connect administrator!"}), 500

    request_times += 1
    logger.info(f"Generate result: {result}")
    return jsonify({"result": result}), 200


@app.route("/character", methods=['POST'])
def generate_character():
    global request_times
    if request_times > 40:
        return jsonify({"result": "The amount of generation is limited daily!"}), 500
    params = request.get_json()
    character1 = params.get('character1')
    character2 = params.get('character2')
    name = params.get('character_name')
    times = params.get('times', 1)

    split_char = '\n###\n'
    length = ((len(character1) + len(character2))/2)
    max_tokens = min(length, 500)
    result = []
    payload = {"max_tokens": max_tokens, "temperature": 0.9, "top_p": 1, "n": 1,
               "stream": False,  'prompt': '', 'suffix': '.', 'stop': ['\n']}
    prompt = character1+split_char+character2+split_char+name
    payload['prompt'] = prompt
    name = name.strip()
    logger.info(f"The prompt : {prompt}")
    for i in range(times):

        response = requests.post(gpt3_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            response_json = response.json()

            text = response_json['choices'][0]["text"]
            logger.info(f"Generate original result: {text}")
            result_ = text.split(split_char)[0]
            result_list = result_.split('.')
            result_ = '.'.join(result_list[:-1])
            logger.info(f"After process: {result_}")
            result.append(name+' '+result_)

    request_times += 1

    return jsonify({"result": result}), 200


@app.route('/autoemotion', methods=['POST'])
def auto_emotion_dialogue():
    global request_times
    if request_times > 40:
        return jsonify({"result": "The amount of generation is limited daily!"}), 500
    not_talk = "This is a chatbot. It can't discuss racial discrimination, gender discrimination, politics and other related issues.\n"
    params = request.get_json()
    prompt = params.get('backstory')
    enneagram = params.get('enneagram')
    scene = params.get('scene')
    dialogue_history = params.get('context')
    turns = params.get('turns', 2)
    times = params.get('times', 1)

    dialogue_history_list = dialogue_history.split('\n')
    bot_name = dialogue_history_list[-1]
    # 不提供初始情绪时，自动标记
    if "(" not in dialogue_history:

        # dialogue emotion label
        marked_sentence = marking_sentence(dialogue_history_list[:-1])
        marked_sentence.append(bot_name)
        dialogue_history = '\n'.join(marked_sentence)

    prompt = not_talk + prompt
    prompt_ = prompt+'\n'+enneagram+'\n'+scene+'\n'+dialogue_history

    payload = {"max_tokens": 200, "temperature": 0.7, "top_p": 1, "n": 1,
               "stream": False,  'prompt': prompt_}
    logger.info(f"Prompt: {prompt_}")
    result = []
    for j in range(times):
        count = 0
        dialog_history_temp = ''
        for i in range(turns):
            response = requests.post(gpt3_dialog_url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                response_json = response.json()

                text = response_json['choices'][0]["text"]

                pattern = re.compile("[\(](.*?)[\)]")
                text = pattern.sub('', text)

                text = text.strip()
                text_list = text.split("\n")

                text_list = marking_sentence(text_list)

                # 拿到前2轮或者前1轮
                texts = pre_process(text_list)

                if i == 0:
                    prompt_temp, dialog_history_temp = process_prompt(prompt, dialogue_history, texts, bot_name)

                else:
                    dialog_history_temp_ = dialogue_history if not dialog_history_temp else dialog_history_temp + '\n' + bot_name
                    prompt_temp, dialog_history_temp = process_prompt(prompt, dialog_history_temp_, texts, bot_name)

                payload['prompt'] = prompt_temp + '\n' + bot_name
                if i == turns-1:
                    result.append(dialog_history_temp)
                    result = [pattern.sub('', t) for t in result]
            else:
                count += 1
                continue

        if count == times:
            return jsonify({"result": "No generate result, if many times, please connect administrator!"}), 500

    request_times += 1
    logger.info(f"Generate result: {result}")
    return jsonify({"result": result}), 200


@app.route('/multiturn_dialogue', methods=['POST'])
def multiturn_dialogue():
    """
    有指代消解的多轮对话
    :return:
    """
    params = request.get_json()
    max_tokens = params.get('max_tokens', 100)
    temperature = params.get('temperature', 1)
    prompt = params.get('prompt')
    top_p = params.get('top_p', 1)
    n = params.get('n', 1)
    input_text = params.get('user_question', '')
    historys = params.get('historys', [])  #[[{'user_name':'...'}, {'node_name':'.....'}], [{'user_name':'...'}, {'node_name':'.....'}], [{'user_name':'...'}, {'node_name':'.....'}],...]
    emotion_historys = params.get('emotion_historys', [])
    node_current_emotion = params.get('node_current_emotion', '')
    user_current_emotion = params.get('user_current_emotion', '')
    user_name = params.get('user_name')
    node_name = params.get('node_name')

    if historys:
        temp_ = []
        for turn in historys:
            for item in turn:
                temp_.extend(list(item.values()))

        historys_sentence = " ## ".join(temp_)

        doc = nlp(historys_sentence)
        if doc._.has_coref:
            mentions = [
                {
                    "start": mention.start_char,
                    "end": mention.end_char,
                    "text": mention.text,
                    "resolved": cluster.main.text,
                }
                for cluster in doc._.coref_clusters
                for mention in cluster.mentions
            ]

            for temp in mentions:

                raw_text = temp['text']
                resolved = temp['resolved']
                if raw_text == resolved:
                    continue
                else:
                    historys_sentence = historys_sentence.replace(raw_text, resolved)
        temp_split = historys_sentence.split(' ## ')
        historys_ = []
        for i in range(len(temp_split), step=2):
            historys_.append([{user_name: temp_split[i], node_name: temp_split[i+1]}])
    else:
        historys_ = historys
    utterance = process_history(historys_, emotion_historys, user_name, node_name)
    prompt = prompt + '\n' + utterance + f'{user_name}: {input_text}\n{node_name}: '

    payload = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "n": n,
               "stream": False,  'prompt': prompt}

    response = requests.post(gpt3_dialog_url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        response_json = response.json()

        text = response_json['choices'][0]["text"]
    else:
        return jsonify({"result": "error"}), 500
    logger.info(f"Generate result: {text}")
    return jsonify({"result": text}), 200


@app.route('/erine3', methods=['POST'])
def erine3():
    import requests
    """
    使用 requests 库发送请求
    使用 pip（或者 pip3）检查我的 python3 环境是否安装了该库，执行命令
      pip freeze | grep requests
    若返回值为空，则安装该库
      pip install requests
    """

    # # 目标文本的 本地文件路径，UTF-8编码，最大长度4096汉字
    # TEXT_FILEPATH = "【您的测试文本地址，例如：./example.txt】"

    # 可选的请求参数
    # top_num: 返回的分类数量，不声明的话默认为 6 个
    params = request.get_json()

    PARAMS = {"top_num": 6}
    PARAMS["text"] = params['prompt']
    # 服务详情 中的 接口地址
    MODEL_API_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/text_gen/daily_dialogue"

    # 调用 API 需要 ACCESS_TOKEN。若已有 ACCESS_TOKEN 则于下方填入该字符串
    # 否则，留空 ACCESS_TOKEN，于下方填入 该模型部署的 API_KEY 以及 SECRET_KEY，会自动申请并显示新 ACCESS_TOKEN
    ACCESS_TOKEN = "24.58cab205aa343ad345bfbe5800b9c4eb.2592000.1654943736.282335-26217326"
    API_KEY = "Xom009EKQpX1Sj21b0Y0ne1c"
    SECRET_KEY = "5b0QBQiGipQAxofXDN2brN9IOaYkZrjG"

    if not ACCESS_TOKEN:
        auth_url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"               "&client_id={}&client_secret={}".format(
            API_KEY, SECRET_KEY)
        auth_resp = requests.get(auth_url)
        auth_resp_json = auth_resp.json()
        ACCESS_TOKEN = auth_resp_json["access_token"]
        print("新 ACCESS_TOKEN: {}".format(ACCESS_TOKEN))
    else:
        print("2. 使用已有 ACCESS_TOKEN")

    request_url = "{}?access_token={}".format(MODEL_API_URL, ACCESS_TOKEN)
    response = requests.post(url=request_url, json=PARAMS)
    if response.status_code == 200:
        response_json = response.json()
        result = response_json['result']['content']
        print(f"result: {response_json['result']}")
        return {"sentence": result, "score": 0.9}
    else:
        print(response.status_code)
        return {"sentence": "", "score": 0}



class Config:
    # cleared every day of 0 hour
    JOBS = [
        {
            'id': 'clear_request_times',
            'func': '__main__:clear_request_times',
            'args': (),
            'trigger': 'cron',
            'day': '*/1',
            'hour': 0,
            # 'minutes': 1
            # 'second': '*/5'
        }

    ]


def clear_request_times():
    global request_times
    request_times = 0
    print("cleard!")


CORS(app, supports_credentials=True)


if __name__ == '__main__':
    app.config.from_object(Config())
    scheduler = APScheduler(BackgroundScheduler(timezone=utc))
    scheduler.init_app(app)
    scheduler.start()
    app.run("0.0.0.0", port=8078, debug=False)

