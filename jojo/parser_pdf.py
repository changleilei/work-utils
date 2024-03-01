# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/4/3 3:36 PM
==================================="""

# import PyPDF2
# import requests
# from tqdm import tqdm


# def parse_pdf(file_path):
#     pdf_file = open(file_path, 'rb')
#     pdf_reader = PyPDF2.PdfReader(pdf_file)
#     page_count = pdf_reader.pages
#     paper_text = []

#     for i in range(len(page_count)):
#         page = page_count[i]

#         # def visitor_body(text, cm, tm, fontDict, fontSize):
#         #     x = tm[4]
#         #     y = tm[5]
#         #     # ignore header/footer
#         #     if (y > 50 and y < 720) and (len(text.strip()) > 1):
#         #         page_text.append({
#         #             'fontsize': fontSize,
#         #             'text': text.strip().replace('\x03', ''),
#         #             'x': x,
#         #             'y': y
#         #         })
#         #
#         # _ = page.extract_text(visitor_text=visitor_body)
#         # blob_font_size = None
#         # blob_text = ''
#         # precessed_text = []
#         # for t in page_text:
#         #     if t['fontsize'] == blob_font_size:
#         #         blob_text += f" {t['text']}"
#         #         if len(blob_text) >= 2000:
#         #             precessed_text.append({
#         #                 'fontsize': blob_font_size,
#         #                 'text': blob_text,
#         #                 'page': i
#         #             })
#         #             blob_font_size = None
#         #             blob_text = ''
#         #     else:
#         #         if blob_font_size is not None and len(blob_text) >= 1:
#         #             precessed_text.append({
#         #                 'fontsize': blob_font_size,
#         #                 'text': blob_text,
#         #                 'page': i
#         #             })
#         #         blob_font_size = t['fontsize']
#         #         blob_text = t['text']
#         #     paper_text += precessed_text
#         paper_text.append(page.extract_text())
#     with open('../data/JOJO/jojo.txt', 'a') as f:
#         f.writelines(paper_text)

# def process_jojo_question():
#     import pandas as pd
#     df = pd.read_csv('../data/JOJO/常见问题FAQ.csv')
#     with open('../data/JOJO/jojo_question.txt', 'w', encoding='utf8') as f2:
#         for q, a in zip(df['问法'], df['参考话术']):
#             if pd.isna(q):
#                 break
#             qa_dict = {'instruction': '', 'output': ''}
#             qa_dict['instruction'] = q.strip()
#             qa_dict['output'] = a.strip()
#             f2.write(str(qa_dict) + '\n')

# def process_jojo_question1():
#     import pandas as pd
#     with open('../data/JOJO/常见问题FAQ.csv') as f,open('../data/JOJO/jojo_question.txt', 'w', encoding='utf8') as f2:
#         lines = f.readlines()[1:]

#         for line in lines:
#             q, a = line.split(',')
#             if pd.isna(q):
#                 break
#             qa_dict = {'instruction': '', 'output': ''}
#             qa_dict['instruction'] = q.strip()
#             qa_dict['output'] = a.strip()
#             f2.write(str(qa_dict) + '\n')

# def generate_question():
#     question_template = "请根据句子中的内容生成的问题：\n\m{}\n问题："
#     payload = {}
#     headers = {
#         'User-Agent': 'apifox/1.0.0 (https://www.apifox.cn)'
#     }
#     url = 'http://10.128.6.19:8088/generate?sentence={}'

#     with open('../data/JOJO/jojo.txt', 'r') as f, open('../data/JOJO/jojo_question.txt', 'a') as f2:
#         text = f.readlines()
#         for i in range(5):
#             for line in tqdm(text):
#                 qa_dict = {'question': '', 'answer': ''}
#                 t = url.format(question_template.format(line))
#                 response = requests.request("POST", t, headers=headers, data=payload)
#                 res = response.text.strip('"')
#                 qa_dict['question'] = res
#                 qa_dict['answer'] = line
#                 f2.write(str(qa_dict) + '\n')


def process_yace_data(data_path):
    import json
    import pandas as pd

 
    df = pd.read_excel(data_path, sheet_name='Sheet1', usecols=['question', 'history'])
    result = {"question": [], "history": []}
# 使用 pandas 读取 Excel 文件的 'Sheet1' 中的 A 和 B 列
    with open("/Users/changlei/Downloads/JoJo_Work_data/datas/性能测试数据集/有效问筛选结果-20240125.txt", 'w', encoding='utf8') as f2:
        count = 0
        for index, row in df.iterrows():   
            # 替换换行符和单引号
            clean_a = str(row['question']).replace("\n", "").replace("'", '"')
            clean_b = str(row['history']).replace("\n", "").replace("'", '"')

            try:
                # 将 clean_b 转换为列表，如果 clean_b 是 '[]'，则保留为空列表
                history_list = json.loads(clean_b) if clean_b != "[]" else []
            except json.JSONDecodeError:
                # 如果 clean_b 不是有效的 JSON 格式，设置为默认值（例如空列表）
                history_list = []

                # 构造字典
            data_dict = {
                "model": "minimax",
                "historys": history_list,
                "knowledge": "",
                "current_input": clean_a,
                "tags": [],
                "categories": []
            }

            # 将字典转换为字符串并打印
            f2.write(json.dumps(data_dict, ensure_ascii=False)+"\n")
            count += 1
        print(count)




if __name__ == '__main__':
    # parse_pdf('../data/品牌介绍（最新版）.pdf')
    # generate_question()
    # process_jojo_question()
    process_yace_data('/Users/changlei/Downloads/JoJo_Work_data/datas/性能测试数据集/有效问筛选结果-20240125.xlsx')
