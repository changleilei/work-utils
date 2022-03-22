# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2021/12/16 10:07 上午
==================================="""
import xml.dom.minidom as xmldom
from lxml import etree
import pandas as pd
from urllib import parse
import json

from tqdm import tqdm

wiki_zh_data_path = "../data/zhwiki-latest-abstract-zh-cn1.xml"
import requests


def clean_wiki_zh_xml():
    dom = xmldom.parse(wiki_zh_data_path)
    root = dom.documentElement
    stop_words = ['参见', '注记', '参考书目', '参考网址', '分支', '主分支', '特殊分支', '外部链接', '参考著作', '选集', '专题介绍', '入门',
                  '扩展阅读', '脚注', '相关列表', '相关学科', '其它', '其他描述型式', '注解', '学科', '参考文献', '参考资料', '参见', '参看']
    docs = root.getElementsByTagName('doc')
    count = 0
    with open('../data/wiki_cleaned.txt', 'w', encoding='utf-8') as f:
        t = 'Wikipedia：'
        for node in docs:
            count += 1
            if node.getElementsByTagName('title')[0].firstChild is not None:
                title = node.getElementsByTagName('title')[0].firstChild.data
            title = title.replace(t, '')
            if node.getElementsByTagName('abstract')[0].firstChild is not None:
                abstract = node.getElementsByTagName('abstract')[0].firstChild.data
            relate_word_links = node.getElementsByTagName('sublink')
            relate_words = []
            for link in relate_word_links:
                if link.firstChild.firstChild is not None:
                    anchor = link.firstChild.firstChild.data
                if anchor not in stop_words:
                    relate_words.append(anchor)
            result = {'title': title, 'abstract': abstract, 'relate_words': relate_words}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"total {count} entites")


def clean_wiki_zh_txt():
    path_ = "../data/wiki_cleaned2.txt"
    save_path = "../data/wiki_cleaned3.txt"
    garbage_data_path = "../data/wiki_garbage.txt"

    import re
    # pattern = re.compile("[（](.*?)[）]")
    pattern = re.compile("[（][）]")
    # 找出abstract中包含英文，数字、标点的作为垃圾数据
    pattern_2 = re.compile("[\{\}\|\]\|\-\=]+")
    # if pattern_2.search("）|计算 (计算机科学)"):
    #     print("True")
    # pattern_2_result = pattern_2.sub("", "）|计算 (计算机科学)")
    with open(path_, 'r', encoding='utf-8') as file, open(save_path, 'w', encoding='utf8') as save_file:
        with open(garbage_data_path, 'w', encoding='utf8') as garbage_data:
            for line in file:
                line_json = json.loads(line)
                abstract = pattern.sub('', line_json['abstract'])
                if pattern_2.search(abstract) or len(abstract) < 3:
                    garbage_data.write(json.dumps(line_json, ensure_ascii=False) + '\n')
                    continue
                line_json['abstract'] = abstract
                save_file.write(json.dumps(line_json, ensure_ascii=False) + '\n')
    # print(pattern_2_result)


def reconstruct_txt2json():
    """将txt转换为.json，内部使用 question， answer作为字段"""
    data_path = "../data/wiki_cleaned3.txt"
    save_path = "../data/wiki_cleaned3.json"
    with open(data_path, 'r', encoding='utf-8') as file, open(save_path, 'w', encoding='utf8') as save_file:
        for line in file:
            line_json = json.loads(line)
            question = line_json['title']
            answer = line_json['abstract']
            save_file.write(json.dumps({"question": question, "answer": answer}, ensure_ascii=False) + '\n')


def reconstruct_csv2json():
    """将txt转换为.json，内部使用 question， answer作为字段"""
    data_path = "../data/qa525998.csv"
    save_path = "../data/qa525998.json"
    datas = pd.read_csv(data_path)
    with open(save_path, 'w', encoding='utf8') as save_file:
        for question, answer in zip(datas['question'], datas['answers']):
            save_file.write(json.dumps({"question": question, "answer": answer}, ensure_ascii=False) + '\n')


def wiki_tongxun_clean():
    data_path = "../data/wiki_cleaned3.json"
    save_path = "../data/wiki_tongshun.json"
    no_path = "../data/wiki_no_tongshun.json"
    server_url = "http://36.189.234.222:60071/z"
    payload = {
        "number": 1,
        "length": 10,
        "max_num": 20,
        "prompt": "",
        "top_p": 0,
        "top_k": 0,
        "temperature": 0.8
    }

    with open(data_path, 'r', encoding='utf8') as input_file, open(save_path, 'a', encoding='utf8') as save_file:
        with open(no_path, 'a', encoding='utf8') as garbage_file:
            prompt = "原句:{0} 标签:"
            for i, line in tqdm(enumerate(input_file)):
                if i < 91272:
                    continue
                answer = json.loads(line, strict=True).get('answer')
                payload['prompt'] = prompt.format(answer)

                response = requests.post(server_url, data=json.dumps(payload))
                if response.status_code == 200:
                    result = response.json().get('result')[0]
                    if result == "通顺<eos>":
                        save_file.write(line)
                    else:
                        garbage_file.write(line)
                else:
                    print(line)


def temp_func():
    data_path = "../data/wiki_cleaned3.json"
    with open(data_path, 'r', encoding='utf8') as input_file:
        for text in input_file.readlines(91272)[:5]:
            print(text)


if __name__ == '__main__':
    # clean_wiki_zh_xml()
    # clean_wiki_zh_txt()
    # reconstruct_txt2json()
    # reconstruct_csv2json()
    wiki_tongxun_clean()
    # temp_func()