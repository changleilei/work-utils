# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/3/22 9:21 上午
==================================="""
import json
import time

import pandas as pd
import random
import threading
import requests
import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

data_dir = '../data/dailydialogue/dialogues_text.txt'
save_dir = '../data/dailydialogue/dialogues.json'


def display():
    with open(data_dir, 'r', encoding='utf8') as f:
        for line in f:
            print(line)


def pre_process():
    result = []
    with open(data_dir, 'r', encoding='utf8') as f, open(save_dir, 'w', encoding='utf8') as s_f:
        for line in f:

            sentences = line.strip().split("__eou__")
            content = {"content": [sentence for sentence in sentences if sentence]}

            result.append(content)
        json.dump(result, s_f, ensure_ascii=False)

def get_qa_results(content_):

    qa_match_url = 'http://39.101.142.52:8082/qa_match'

    payload_qa = {'questions': []}

    payload_qa['questions'].extend(content_)
    try:
        response = requests.post(qa_match_url, data=json.dumps(payload_qa))
        if response.status_code == 200:
            result = response.json()['result']
            score = result[0][0]
            return score
        else:
            print(response.status_code)
            print(content_)
            return 0.0
    except Exception as e:
        print(content_)
        print(e)
        return 0.0


def get_sgpt_result(content_):
    # sgpt_model_url = 'http://36.189.234.222:60033/z/ranking/sgpt'
    sgpt_model_url = 'http://47.104.129.11:8081/z/ranking/score'
    payload_sgpt = {'texts': []}
    headers = {
        'User-Agent': 'Apipost client Runtime/+https://www.apipost.cn/',
        'Content-Type': 'application/json',
    }
    payload_sgpt['texts'].extend(content_)
    try:
        response = requests.post(sgpt_model_url, headers=headers, data=json.dumps(payload_sgpt))
        if response.status_code == 200:
            result = response.json()['result'][0]['score']
            return result
        else:
            print(response.status_code)
            print(content_)
            return 0.0
    except Exception as e:
        print(content_)
        print(e)
        return 0.0


def get_qamatch_result(content_):
    # sgpt_model_url = 'http://36.189.234.222:60033/z/ranking/sgpt'
    qamatch_model_url = 'http://47.104.129.11:8074/ranking/cross_score'
    payload_sgpt = {'texts': []}
    headers = {
        'User-Agent': 'Apipost client Runtime/+https://www.apipost.cn/',
        'Content-Type': 'application/json',
    }
    payload_sgpt['texts'].extend(content_)
    try:
        response = requests.post(qamatch_model_url, headers=headers, data=json.dumps(payload_sgpt))
        if response.status_code == 200:
            result = response.json()['result'][0]['score']
            return result
        else:
            print(response.status_code)
            print(content_)
            return 0.0
    except Exception as e:
        print(content_)
        print(e)
        return 0.0

def current_score():

    result = {'question': [], 'answer': [], 'qa_match_score': [], 'sgbt_score': []}

    with open(save_dir, 'r', encoding='utf8') as f:
        datas = json.load(f)  # datas length 13118
        for data in tqdm.tqdm(datas):
            content = data['content']
            if len(content) == 2:
                question = content[0]
                answer = content[1]
                qa_match_score = round(get_qa_results(content), 4)
                sgbt_score = round(get_sgpt_result(content), 4)
                result['question'].append(question)
                result['answer'].append(answer)
                result['qa_match_score'].append(qa_match_score)
                result['sgbt_score'].append(sgbt_score)
            else:
                for i in range(0, len(content), 2):
                    temp = content[i: i+2]
                    if len(temp) < 2:
                        continue
                    question = temp[0]
                    answer = temp[1]
                    qa_match_score = round(get_qa_results(temp), 4)
                    sgbt_score = round(get_sgpt_result(temp), 4)
                    result['question'].append(question)
                    result['answer'].append(answer)
                    result['qa_match_score'].append(qa_match_score)
                    result['sgbt_score'].append(sgbt_score)

        frame = pd.DataFrame(result)
        frame.to_csv('../data/dailydialogue/dailydilogue_score.csv')


class Mythread(threading.Thread):
    def __init__(self, content_):
        super(Mythread, self).__init__()
        self.content = content_
        self.result = {'question': [], 'answer': [], 'qa_match_score': [], 'sgbt_score': []}

    def run(self):
        for data in tqdm.tqdm(self.content):
            content = data['content']
            if len(content) == 2:
                question = content[0]
                answer = content[1]
                qa_match_score = round(get_qa_results(content), 4)
                sgbt_score = round(get_sgpt_result(content), 4)
                self.result['question'].append(question)
                self.result['answer'].append(answer)
                self.result['qa_match_score'].append(qa_match_score)
                self.result['sgbt_score'].append(sgbt_score)
            else:
                for i in range(0, len(content), 2):
                    temp = content[i: i+2]
                    if len(temp) < 2:
                        continue
                    question = temp[0]
                    answer = temp[1]
                    qa_match_score = round(get_qa_results(temp), 4)
                    sgbt_score = round(get_sgpt_result(temp), 4)
                    self.result['question'].append(question)
                    self.result['answer'].append(answer)
                    self.result['qa_match_score'].append(qa_match_score)
                    self.result['sgbt_score'].append(sgbt_score)

    def get_result(self):
        return self.result


def current_score_thread():
    task_list = []
    result_ = {'question': [], 'answer': [], 'qa_match_score': [], 'sgbt_score': []}
    with open(save_dir, 'r', encoding='utf8') as f:
        datas = json.load(f)  # datas length 13118
        for i in range(0, len(datas), 3000):
            t = Mythread(datas[i: i+3000])
            task_list.append(t)
        for t in task_list:
            t.start()
            t.join()
        while len(task_list) != 0:
            for t in task_list:
                if not t.is_alive():
                    result = t.get_result()
                    if len(result['question']) > 0:
                        result_['question'].extend(result['question'])
                        result_['answer'].extend(result['answer'])
                        result_['qa_match_score'].extend(result['qa_match_score'])
                        result_['sgbt_score'].extend(result['sgbt_score'])
                    else:
                        print(result)
                    task_list.remove(t)
        frame = pd.DataFrame(result_)
        frame.to_csv('../data/dailydialogue/dailydilogue_score.csv')


def analysis():

    datas = pd.read_csv('../data/dailydialogue/dailydilogue_score.csv')
    print(datas['qa_match_score'].value_counts())
    print(datas['sgbt_score'].value_counts())


def generate_sts_dataset():
    result_ = {'questions': [], 'answers': [], 'qa_match': []}
    # negative_samples = []
    # with open(save_dir, 'r', encoding='utf8') as f:
    #     datas = json.load(f)
    #     for data in datas:
    #         content = data['content']
    #         if len(content) == 2:
    #             negative_sam = content[1].strip()
    #             negative_samples.append(negative_sam)
    #         else:
    #             for i in range(0, len(content), 2):
    #                 temp = content[i: i+2]
    #                 if len(temp) < 2:
    #                     continue
    #                 negative_sam = temp[1].strip()
    #                 negative_samples.append(negative_sam)
    frame = pd.read_csv('../data/dailydialogue/dailydilogue_score.csv', index_col=[0])
    for question, answer in zip(frame['question'], frame['answer']):
        question = question.strip()
        answer = answer.strip()
        result_['questions'].append(question)
        result_['answers'].append(answer)
        result_['qa_match'].append(1)

        # negative = random.choice(negative_samples)
        # while negative != answer:
        #     result_['questions'].append(question)
        #     result_['answers'].append(negative)
        #     result_['qa_match'].append(0)
        #     break

    pd.DataFrame(result_).to_csv('../data/dailydialogue/dailydilogue_qamatch.csv')


def shuffle_qamatch():
    frame = pd.read_csv('../data/dailydialogue/dailydilogue_qamatch.csv')
    # frame_list = frame
    train_data, test_data = train_test_split(frame, train_size=0.8, shuffle=True, random_state=66)
    pd.DataFrame(train_data).to_csv('../data/dailydialogue/dailydilogue_qamatch_train.csv', index=False)
    pd.DataFrame(test_data).to_csv('../data/dailydialogue/dailydilogue_qamatch_test.csv', index=False)


class Mythread2(threading.Thread):
    def __init__(self, content_):
        super(Mythread2, self).__init__()
        self.content = content_
        self.result = {'question': [], 'answer': [], 'qa_match_score': []}

    def run(self):
        for question, answer in tqdm.tqdm(zip(self.content['question'], self.content['answer'])):
            self.result['question'] = question
            self.result['answer'] = answer
            self.result['qa_match_score'] = round(get_qamatch_result([question, answer]),4)
            time.sleep(1)

    def get_result(self):
        return self.result


def singal_score():
    data_path = '../data/qa_match_cn/qa_match_cn.csv'
    frame = pd.read_csv(data_path)
    # frame['new_qa_match_score'] = [0] * len(frame)
    datas = frame[['question', 'answer']]
    task_list = []
    for i in range(0, len(datas), 10000):
        data = datas[i: i+10000]

        t = Mythread2(data)
        task_list.append(t)
    for t in task_list:
        t.start()
    scores = []
    all_results = {}
    while len(task_list) != 0:
        for t in task_list:
            if not t.is_alive():
                result = t.get_result()
                scores.extend(result['qa_match_score'])
                all_results['question'].extend(result['question'])
                all_results['answer'].extend(result['answer'])
                all_results['qa_match_score'].extend(result['qa_match_score'])

    pd.DataFrame(all_results).to_csv('../data/qa_match_cn/qa_match_score.csv')
    # frame['new_qa_match_score'] = scores
    # frame.to_csv('../data/dailydialogue/dailydilogue_score_new.csv')


def get_result():
    data_path = '../data/qa_match_cn/qa_match_1.5w_sub0601.csv'
    frame = pd.read_csv(data_path)
    # frame['new_qa_match_score'] = [0] * len(frame)
    result = {'question': [], 'answer': [], 'reranking_score': []}
    datas = frame[['question', 'answer', 'label']]
    for question, answer in tqdm.tqdm(zip(datas['question'], datas['answer'])):
        result['question'].append(question)
        result['answer'].append(answer)
        result['reranking_score'].append(round(get_qamatch_result([question, answer]), 4))
    datas['reranking_score'] = result['reranking_score']
    # pd.DataFrame(result).to_csv('../data/qa_match_cn/qa_match_score.csv')
    datas.to_csv('../data/qa_match_cn/1.5w_scores2.csv')


def pre_process2():
    result = []
    with open(data_dir, 'r', encoding='utf8') as f, open(save_dir, 'w', encoding='utf8') as s_f:
        for line in f:

            sentences = line.strip().split("__eou__")
            sentences = [sentence.strip() for sentence in sentences]
            result.append('\n'.join(sentences))
    with open('../data/dailydialogue/dailydilogue_amr.txt', 'w', encoding='utf8') as f:
        f.writelines(result)
if __name__ == '__main__':
    # display()
    # pre_process()
    # current_score()
    # analysis()
    # generate_sts_dataset()
    # shuffle_qamatch()
    # singal_score()
    get_result()
    # pre_process2()