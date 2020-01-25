#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 打开文件
with open("600entity.txt", "r") as f:
    result = []
    for j in f.readlines():
        result.append([j.split()[0], j.split()[-1], j.split()[-3], j.split()[-2]])
    result = pd.DataFrame(result[1::], columns=result[0])
    categories = set(result["calss"].values)

with open("original_600.txt", "r") as f:
    data = [sentences[:-1] for sentences in f.readlines()]
    data.pop(-1)


def biaozhu_data(original, entity):
    label = [["O"] * len(i) for i in original]
    mapentity = {"独立症状": "DES", '症状描述': 'SYD', '疾病': 'DIS', '药物': 'MED', '手术': 'SUR', '解剖部位': 'ANA'}
    for index, row in entity.iterrows():
        label[int(row["case_id"])][int(row["begin"])] = "B-" + mapentity[row["calss"]]
        for i in range(int(row["begin"]) + 1, int(row["end"]), 1):
            label[int(row["case_id"])][i] = "I-" + mapentity[row["calss"]]
    X_train, X_test, y_train, y_test = train_test_split(
        original, label, test_size=0.1, random_state=91)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.1, random_state=918)
    return [X_train, X_test, y_train, y_test, X_dev, y_dev]


def save_txt(dir, source):
    out = []
    for char in source:
        out.append("|".join(char) + "\n")
    with open(dir, "w") as f:
        f.writelines(out)


if __name__ == "__main__":
    out = biaozhu_data(data, result)
    save_txt("tmp"+ "/source.txt", out[0])
    save_txt("tmp" + "/test.txt", out[1])
    save_txt("tmp" + "/target.txt", out[2])
    save_txt("tmp"  + "/test_target.txt", out[3])
    save_txt("tmp"  + "/dev.txt", out[4])
    save_txt("tmp"  + "/dev_target.txt", out[5])
