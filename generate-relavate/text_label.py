# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2021/12/10 4:58 下午
==================================="""
import time

import requests
from tqdm import tqdm
import threading
import multiprocessing
data_path = "../data/STC.json"
test_path = "../data/STC_test.json"

import json
import pandas as pd

headers = {
    'Content-Type': 'application/json'
}

label_url = "http://39.101.149.45:8083/z"
ner_url = " http://192.168.0.181:8088/ner"

label_2_zh_path = "../data/cluener_label.json"
zh_2_en_labels = json.load(open(label_2_zh_path, 'r', encoding='utf8'))["label_list"]
en_2_zh_labels = {value: key for key, value in zh_2_en_labels.items()}

payload_label = {
    "number": 1,
    "length": 1000,
    "max_num": 50,
    "prompt": "",
    "top_p": 0.9,
    "top_k": 30,
    "temperature": 0.9
}

prompt = "文本分类：\n {} \n选项：教育，三农，娱乐，健康，搞笑，美食，财经，科技，旅游，汽车，房产，母婴，" \
         "体育，国际，宠物，游戏，职场，艺术，动漫\n答案："

data_dir = "../data/stc_4412427chitchat.csv"

frame = pd.read_csv(data_dir)

def func():
    result = {'questions': [], "answers": []}
    all_data = []
    with open(data_path, 'r', encoding='utf8') as da, open(test_path, 'r', encoding='utf8') as te:
        data_train = json.load(da)
        data_test = json.load(te)
        all_data.extend(data_train['train'])
        all_data.extend(data_test['test'])
    conut_turn = 0
    for list_ in all_data:
        questions = ""
        answers = ""
        for i, line in enumerate(list_):
            if i < 1:
                questions = "".join(line.split(" "))
            else:
                answers = "".join(line.split(" "))

        result['questions'].append(questions)
        result['answers'].append(answers)
    print(conut_turn)
    frame = pd.DataFrame(result)
    frame.to_csv(f"../data/stc_{len(result['questions'])}chitchat.csv")


def label_and_ner():

    result = {"question": [], 'answer': [], 'label': [], 'ners': []}

    i = 0
    for question, answer in tqdm(zip(frame["questions"][297128:], frame["answers"][297128:]), desc="label and ner"):
        prompt_ = prompt.format(question+answer)
        payload_ner = {
            'text': [question+answer]
        }

        payload_label['prompt'] = prompt_
        try:
            response_label = requests.request('POST', label_url, headers=headers, data=json.dumps(payload_label))
            response_ner = requests.request('POST', ner_url, headers=headers, data=json.dumps(payload_ner))
            label_list = response_label.json()['result']
            if len(label_list) < 1:
                label = ""
            else:
                label = label_list[0]
            result["label"].append(label)
            ner = response_ner.json()  # [[]]
            i += 1
            for tags in ner:
                ners = []

                for tag in tags:
                    ners.append({en_2_zh_labels[tag['ner']]: tag['word']})

                result["ners"].append(ners)
            result["question"].append(question)
            result["answer"].append(answer)

        except Exception as e:
            print(f"error was happen, {e}")
            print(f"current process id is {i}")
        if i % 50000 == 0:
            datas_frame = pd.DataFrame(result)
            datas_frame.to_csv(f"../data/stc_labeled_{(i/50000)*5}w.csv", index=False)
            print(f"save success, already processed {i} items ")
            for key in result.keys():
                result[key] = []


def process_func(question, answer):
    result = {"question": [], 'answer': [], 'label': [], 'ner_label': [], 'ner_name': []}

    prompt_ = prompt.format(question+answer)
    payload_ner = {
        'text': [question+answer]
    }

    payload_label['prompt'] = prompt_
    result['question'].append(question)
    result['answer'].append(answer)
    try:
        response_label = requests.request('POST', label_url, headers=headers, data=json.dumps(payload_label))
        response_ner = requests.request('POST', ner_url, headers=headers, data=json.dumps(payload_ner))
        label_list = response_label.json()['result']
        if len(label_list) < 1:
            label = ""
        else:
            label = label_list[0]
        result["label"].append(label)
        ner = response_ner.json()  # [[]]]
        for tags in ner:
            ner_labels = []
            ner_names = []

            for tag in tags:
                label = en_2_zh_labels[tag["ner"]]
                ner_labels.append(label)
                ner_names.append(tag['word'])
            if len(ner_labels) < 1:
                result["ner_label"].append("")
                result["ner_name"].append("")
            else:
                result["ner_label"].append("&&".join(ner_labels))
                result["ner_name"].append("&&".join(ner_names))

    except Exception as e:
        print(f"error was happen, {e}")
    return result


def multi_threading_func(worker=10):

    result = {"question": [], 'answer': [], 'label': [], 'ner_label': [], 'ner_name': []}
    pooler = multiprocessing.Pool(worker)
    result_list = []
    # process_result = pooler.imap(process_func, frame, 25)

    proc_start = time.time()
    i = 0
    for question, answer in tqdm(zip(frame['questions'], frame['answers'])):
        res = pooler.apply_async(process_func, args=(question, answer)).get()
        result['question'].extend(res['question'])
        result['answer'].extend(res['answer'])
        result['label'].extend(res['label'])
        result['ner_label'].extend(res['ner_label'])
        result['ner_name'].extend(res['ner_name'])
        i += 1

        if i % 10000 == 0:
            current = time.time()
            elapsed = current - proc_start
            print(f"processed {i} lines",
                  f"{i/elapsed} lines/s")
            print(f"result length: {len(result['question'])}")
    frame2 = pd.DataFrame(result)
    frame2.to_csv("../data/stc_labeled.csv")


class myThread(threading.Thread):
    def __init__(self, start=0, end=10, thread=0):
        threading.Thread.__init__(self)
        self.start_ = start
        self.end_ = end
        self.thread = thread

    def run(self):

        frame_q_data = frame['questions'][self.start_:self.end_]
        frame_a_data = frame['answers'][self.start_:self.end_]
        result = {"question": [], 'answer': [], 'label': [], 'ner_label': [], 'ner_name': []}
        i = 0
        for question, answer in tqdm(zip(frame_q_data, frame_a_data), desc=f"current thread:{self.thread}"):
            prompt_ = prompt.format(question + answer)
            payload_ner = {
                'text': [question + answer]
            }

            payload_label['prompt'] = prompt_
            try:
                response_label = requests.request('POST', label_url, headers=headers, data=json.dumps(payload_label))
                response_ner = requests.request('POST', ner_url, headers=headers, data=json.dumps(payload_ner))
                label_list = response_label.json()['result']
                if len(label_list) < 1:
                    label = ""
                else:
                    label = label_list[0]
                result["label"].append(label)
                ner = response_ner.json()  # [[]]]
                i += 1
                for tags in ner:
                    ner_labels = []
                    ner_names = []

                    for tag in tags:
                        label = en_2_zh_labels[tag["ner"]]
                        ner_labels.append(label)
                        ner_names.append(tag['word'])
                    if len(ner_labels) < 1:
                        result["ner_label"].append("")
                        result["ner_name"].append("")
                    else:
                        result["ner_label"].append("&&".join(ner_labels))
                        result["ner_name"].append("&&".join(ner_names))
            except Exception as e:
                print(f"error was happen, {e}")
            if i == 0 or i % 100000 == 0:
                print(f"show data question：{result['question'][i]}",
                      f"answer: {result['answer'][i]}",
                      f"label: {result['label'][i]}",
                      f"ner_label: {result['ner_label'][i]}",
                      f"ner_name: {result['ner_name'][i]}")
        frame2 = pd.DataFrame(result)
        frame2.to_csv(f"../data/stc_labeled_{self.start_}-{self.end_}.csv")

def thread_func(args=10):
    threads = []
    j = 0
    t = 4412427
    for i in range(args):
        start_ = i*500000
        end_ = start_ + 500000
        if end_ > t:
            end_ = t
        thread_1 = myThread(start=start_, end=end_, thread=j)
        thread_1.start()
        threads.append(thread_1)
        j += 1

    for t in threads:
        t.join()



if __name__ == '__main__':
    # func()
    label_and_ner()
    # multi_threading_func(1)
    # thread_func(10)