# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/8/12 1:56 下午
==================================="""
import os
import json
import pandas as pd

def ner_data_collections():
    data_path = '../data/DuLeMon/self/'
    files = os.listdir(data_path)

    for fi in files:
        save_filename = 'ner_' + fi
        result = []
        with open(data_path + fi, 'r', encoding='utf8') as f:
            for line in f:
                data = json.loads(line)
                persona = []
                persona.extend(data['p1_persona'])
                persona.extend(data['p2_persona'])
                persona = [''.join(t.split(' ')) for t in persona]
                for text in persona:
                    result.append(json.dumps({"text": text.split(':')[-1],"label":[]}, ensure_ascii=False)+'\n')

        with open(data_path + save_filename, 'w', encoding='utf8') as f:
            f.writelines(result)

def generate_mask_data():

    data_path = '../data/multi_turn/dialog_zh.json'
    result = {'question':[], 'answer':[], 'context':[]}

    with open(data_path, 'r', encoding='utf8') as f:
        datas = json.load(f)

        for i, data in enumerate(datas):
            if i > 10 :
                break
            content = data['content']
            history = ''

            for i in range(0, len(content), 2):
                question = content[i]
                answer = content[i+1]
                result['question'].append(question)
                result['answer'].append(answer)
                result['context'].append(history)
                history += question
                history += answer
    frame = pd.DataFrame(result)
    frame.to_csv('../data/multi_turn/temp_multi.csv', index=False)


def get_labeled_data():
    data_path = '../data/DuLeMon/self/all.jsonl'
    save_path = '../data/DuLeMon/self/labeled.jsonl'

    with open(data_path, 'r', encoding='utf8') as open_f, open(save_path, 'w', encoding='utf8') as save_f:
        for line in open_f:
            line_json = json.loads(line)
            if line_json['label']:
                save_f.write(line)


if __name__ == '__main__':
    get_labeled_data()