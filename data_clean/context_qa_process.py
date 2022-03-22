# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/1/27 4:55 下午
==================================="""
"""用来处理上下文QA数据"""

import pandas as pd


data_path = '../data/qa.txt'


def func():
    result = {"question": [], "answers": []}


    with open(data_path, 'r', encoding="utf8") as f:
        for i, data in enumerate(f.readlines()):

            if data.startswith("问题:"):
                result["question"].append(data.strip("问题:").strip())
            elif data.startswith("回复:"):
                result["answers"].append(data.strip("回复:").strip())
    pre_question = ""
    pre_ans = ""
    result__ = {"question": [], "answers": []}
    for question, ans in zip(result["question"], result["answers"]):
        if pre_question == question and pre_ans == ans:
            continue
        else:
            pre_question = question
            pre_ans = ans
            result__['question'].append(question)
            result__['answers'].append(ans)

    frame = pd.DataFrame(result__)
    frame.to_csv("qa.csv", index=False)
    print(len(result__['question']))


if __name__ == '__main__':
    func()