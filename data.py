#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd

# 打开文件
with open("600entity.txt", "r") as f:
    # file = [word for i in [j.split() for j in f.readlines()] for word in i]
    result = []
    for j in f.readlines():
        result.append([j.split()[0], j.split()[-1], j.split()[-3], j.split()[-2]])
    result = pd.DataFrame(result[1::], columns=result[0])
    categories = set(result["calss"].values)


with open("original_600.txt", "r") as f:
    data = [sentences for sentences in f.readlines()]

# def transe_ner():
