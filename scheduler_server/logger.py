# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/3/1 3:02 下午
==================================="""
import logging
import sys


def get_logger(name, filename='generate.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s ｜ %(levelname)s ｜ %(filename)s ｜ %(funcName)s ｜ %(lineno)s ｜ %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    fh = logging.FileHandler(filename=f'logs/{filename}', encoding='utf-8', mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(stream_handler)
    logger.addHandler(fh)

    return logger
