# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/8/29 6:24 下午
==================================="""
import json
import pandas as pd

def process_zhuti():
    data_path = '../data/zhuti/out_zhuti.txt'
    save_path = '../data/zhuti/out_zhuti_prompt_part.txt'

    template = "{0}\n这句话的主题是: 《{1}》"
    with open(data_path, 'r', encoding='utf8') as f, open(save_path, 'w', encoding='utf8') as w_f:
        first_sentence = ''
        theme = ''
        for i, line in enumerate(f.readlines()):
            if i > 50000:
                break
            if line.startswith('问:'):
                first_sentence = line.replace('问:', '')
            elif line.startswith('主题是:'):
                theme = line.replace('主题是:', '')
            elif len(line.strip()) == 0:
                template_ = template.format(first_sentence, theme.strip())
                w_f.write(json.dumps(template_, ensure_ascii=False) + '\n')
                first_sentence = ''
                theme = ''
            else:
                print(line)


def theme_mapping():
    data_path = '../data/zhuti/out_zhuti.txt'
    save_path = '../data/zhuti/out_zhuti_0923.txt'
    theme_mapping_path = '../data/zhuti/theme.csv'

    theme_datas = pd.read_csv(theme_mapping_path)
    theme_map = {}
    theme_set = set()
    for raw_theme, new_theme in zip(theme_datas['text'], theme_datas['theme']):
        theme_map[raw_theme] = new_theme
        theme_set.add(new_theme)

    # print(len(theme_set))  #94

    with open(data_path, 'r', encoding='utf8') as f, open(save_path, 'w', encoding='utf8') as w_f:
        first_sentence = ''
        theme = ''
        for i, line in enumerate(f.readlines()):
            # 如果使用全部的数据，请把这行注释
            if i > 50000:
                break
            if line.startswith('问:'):
                first_sentence = line.replace('问:', '')
            elif line.startswith('主题是:'):
                theme = line.replace('主题是:', '').strip()
                themes = theme.split('、')

                new_themes = list(set([theme_map[key] for key in themes]))
                theme = '、'.join(new_themes)
            elif len(line.strip()) == 0:
                template_ = fr"{first_sentence.strip()} 这句话的主题是: {theme.strip()}"
                # template_ = template.format(first_sentence, theme.strip())
                w_f.write(template_+ '\n')
                first_sentence = ''
                theme = ''
            else:
                print(line)

def to_bloom_train_format():
    data_path = '../data/zhuti/out_zhuti_0920.txt'
    save_path = '../data/zhuti/all_theme_intent.jsonl'
    with open(data_path, 'r', encoding='utf8') as f, open(save_path, 'w', encoding='utf8') as w_f:
        for line in f:
            texts = {'text': line.strip()}
            w_f.write(json.dumps(texts, ensure_ascii=False)+'\n')


def get_themes():
    data_path = '../data/zhuti/theme.csv'
    data = pd.read_csv(data_path)
    d = list(set(list(data['theme'])))
    with open('../data/zhuti/theme.json', 'w') as f:
        json.dump(d,f, ensure_ascii=False)
        print(d)
        print(len(d))

if __name__ == '__main__':
    # process_zhuti()
    theme_mapping()
    # get_themes()
    # to_bloom_train_format()