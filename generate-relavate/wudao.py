# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/2/25 11:56 上午
==================================="""
from wudaoai import content

engine = "chat"
# engine = "qa"
req_key = "9897ed5856c44a6a9d93c86261bed0c6"
context = {"prompt": "书有什么好看的？"}
content.send(engine, req_key, context)
