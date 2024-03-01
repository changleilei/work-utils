# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/11/15 6:20 下午
==================================="""
import re


def judge_zh(sentence):
    # 判断中文
    result = ''.join(re.findall(re.compile("([\u4E00-\u9FA5]+)"), sentence))
    if len(result) > len(sentence) / 2:
        return True
    else:
        return False

def judge_en(sentence):
    # 判断英文
    ignores = []
    result = ''.join(re.findall(r'[A-Za-z]', sentence))
    result = result.lower()
    for ignore in ignores:
        if ignore in result:
            result = result.replace(ignore, "")
    if len(result) > len(sentence) / 2:
        return True
    else:
        return False


def split_PDF(file_path, split_num=10):
    import PyPDF2

    # 打开PDF文件
    pdf_file = open(file_path, 'rb')

    # 创建PDF文件读取器对象
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # 获取PDF文件中的页数
    page_count = pdf_reader.pages
    a = [1,2,3,4,5,6,7,8]
    b = a[::3]
    for i in range(0, len(page_count), split_num):
        pages = page_count[i:i + split_num]
        pdf_writer = PyPDF2.PdfWriter()
        for page in pages:
            pdf_writer.add_page(page)
        split(pdf_writer, i)

def split(pdf_writer, page):
    with open(f'page_{page + 1}.pdf', 'wb') as new_pdf_file:
        pdf_writer.write(new_pdf_file)
        pdf_writer.close()



if __name__ == '__main__':
    split_PDF('data/品牌介绍（最新版）.pdf', split_num=20)
