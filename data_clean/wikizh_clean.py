# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2021/12/16 10:07 上午
==================================="""
import xml.dom.minidom as xmldom
from lxml import etree

from urllib import parse
import json
wiki_zh_data_path = "../data/zhwiki-latest-abstract-zh-cn1.xml"


def clean_wiki_zh_xml():


    dom = xmldom.parse(wiki_zh_data_path)
    root = dom.documentElement
    stop_words = ['参见', '注记', '参考书目', '参考网址', '分支', '主分支','特殊分支', '外部链接', '参考著作', '选集', '专题介绍', '入门',
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
            f.write(json.dumps(result, ensure_ascii=False)+'\n')

        print(f"total {count} entites")


if __name__ == '__main__':
    clean_wiki_zh_xml()