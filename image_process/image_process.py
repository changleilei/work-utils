# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/6/16 2:24 下午
==================================="""
import json
import glob
import os
from shutil import copyfile
from paddleocr import PaddleOCR, draw_ocr

import pandas as pd
from tqdm import tqdm

def image_compare():
    images_path = '../data/images/*'
    images = glob.glob(images_path)
    json_urls_path = '../data/json_urls/biaoqing.jsonl'
    result = []
    with open(json_urls_path, 'r') as f:
        temp_path = '../data/images/'
        count = 0
        for item in f:
            datas = json.loads(item)
            images_key = datas['ImageKey']
            t = temp_path + images_key
            if t in images:
                datas['local_path'] = images_key
                result.append(json.dumps(datas, ensure_ascii=False)+'\n')
                count += 1
    print(f'count: {count}')
    images_with_local_path = '../data/images_with_local_path/urls.txt'
    with open(images_with_local_path, 'w', encoding='utf8') as f:
        f.writelines(result)


def extract_text_from_images():

    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    # to switch the language model in order.

    out = {"image_names": [], "text": []}
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # need to run only once to download and load model into memory
    images_path = r'../data/images/'
    image_names = os.listdir(images_path)
    for image_name in tqdm(image_names):
        img_path = os.path.join(images_path, image_name)
        result = ocr.ocr(img_path, cls=True)
        print(result)
        out["image_names"].append(image_name.split(".")[0])
        if result:
            all_text = [i[1][0] for i in result]
            out["text"].append("".join(all_text))
        else:
            out["text"].append("")

    out = pd.DataFrame(out)
    out.to_csv("../data/images_with_local_path/out.csv")


def copy_image():
    data_path = '../data/json_urls/project-65.csv'
    images_path = '/data/cll/work-utils/data/json_urls/cleaned/'
    cleaned_data = '/data/cll/work-utils/data/json_urls/cleaned1/'
    images = glob.glob(images_path+'*jpg')

    datas_frame = pd.read_csv(data_path)
    no_water_mark = datas_frame[datas_frame['是否有水印'] == '无水印']
    no_ip = no_water_mark[datas_frame['版权IP'] == '无IP'][datas_frame['choice2'] != '其他']
    for name in tqdm(no_ip['ocr']):
        image_name = images_path+name.split('/')[-1]
        if image_name in images:
            try:
                copyfile(image_name, cleaned_data+name.split('/')[-1])
            except IOError as e:
                print(e)
                exit(1)


def extract_text_from_images2():

    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    # to switch the language model in order.
    data_path = '../data/json_urls/project-65.csv'

    cleaned_data = '/data/cll/work-utils/data/json_urls/cleaned/'
    images = glob.glob(cleaned_data+'*jpg')

    datas_frame = pd.read_csv(data_path)
    no_water_mark = datas_frame[datas_frame['是否有水印'] == '无水印']
    no_ip = no_water_mark[datas_frame['版权IP'] == '无IP']
    out = {"image_path": [], "text": [], 'tag': []}

    for name, word in tqdm(zip(no_ip['ocr'], no_ip['是否有文字'])):
        image_name = cleaned_data+name.split('/')[-1]
        if image_name in images:

            if word == '有字':
                ocr = PaddleOCR(use_angle_cls=True, lang='ch')

                result = ocr.ocr(image_name, cls=True)
                print(result)
                out["image_path"].append(image_name)
                if result:
                    all_text = [i[1][0] for i in result]
                    out["text"].append("".join(all_text))
                else:
                    out["text"].append("")
            else:
                out["image_path"].append(image_name)
                out["text"].append("")

    out = pd.DataFrame(out)
    out.to_csv("../data/images_with_local_path/out1.csv")

def process_tag():
    cleaned_data = '/data/cll/work-utils/data/json_urls/cleaned/'
    data_path = '../data/images_with_local_path/out1.csv'
    raw_data_path = '../data/json_urls/project-65.csv'

    datas_frame = pd.read_csv(raw_data_path)
    no_water_mark = datas_frame[datas_frame['是否有水印'] == '无水印']
    no_ip = no_water_mark[datas_frame['版权IP'] == '无IP']
    out = {"image_name": [], "text": [], "tag": []}
    file_names = {x.split('/')[-1]: tag for x, tag in zip(no_ip['ocr'], no_ip['choice2'])}
    tagsss = set()

    cleaned_data_frame = pd.read_csv(data_path)
    for name, word in tqdm(zip(cleaned_data_frame['image_path'], cleaned_data_frame['text'])):
        file_name = name.split('/')[-1]

        if file_name in file_names:
            tags = file_names.get(file_name)
            if isinstance(tags, float):
                continue
            if len(tags) > 0:
                if '{' in tags:
                    tags = eval(tags)
                if isinstance(tags, str):
                    if tags =='其他':
                        continue
                    tagsss.add(tags)
                    out['image_name'].append(file_name)
                    out['text'].append(word)
                    out['tag'].append(tags)
                elif isinstance(tags, dict):
                    t = tags['choices']
                    for temp in t:
                        if temp == '其他':
                            continue
                        tagsss.add(temp)
                        out['image_name'].append(file_name)
                        out['text'].append(word)
                        out['tag'].append(temp)
                else:
                    continue
    pd.DataFrame(out).to_csv('../data/images_with_local_path/with_tag.csv', index=False)
    print(list(tagsss))


if __name__ == '__main__':
    # image_compare()
    process_tag()