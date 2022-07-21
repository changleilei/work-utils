# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/3/1 3:02 下午
==================================="""
import logging
import sys


def get_logger():
    logger = logging.getLogger('gpt dialogue')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s ｜ %(levelname)s ｜ %(filename)s ｜ %(funcName)s ｜ %(lineno)s ｜ %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)

    return logger
