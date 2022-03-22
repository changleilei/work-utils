# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/2/17 2:59 下午
==================================="""

import random
import numpy as np

temp_list = [x for x in range(130)]


def func1():
    a = []
    for i, x in enumerate(temp_list, 1):
        a.append(x)
        if i % 100 == 0:
            yield a
            a = []
    if len(a) != 0:
        yield a


def func2():
    for a in func1():
        print(a)


if __name__ == '__main__':
    func2()