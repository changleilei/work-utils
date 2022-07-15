# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/7/13 11:09 上午
==================================="""
import pandas
import torch
from sentence_transformers import SentenceTransformer, util

sentence_model = SentenceTransformer("stsb-xlm-r-multilingual")
data_frame = pandas.read_csv("../data/train_all.csv")
fight_list = []
diss_list = []
other_list = []
for row in data_frame.itertuples(index=False):
    if row[0] == 1:
        fight_list.append(str(row[1]))
    elif row[0] == 2:
        diss_list.append(str(row[1]))
    else:
        other_list.append(str(row[1]))


document_type_1 = sentence_model.encode(fight_list, convert_to_tensor=True)
document_type_2 = sentence_model.encode(diss_list, convert_to_tensor=True)
document_type_3 = sentence_model.encode(other_list, convert_to_tensor=True)
documents = [document_type_1,document_type_2,document_type_3]
text_pool = [fight_list, diss_list, other_list]
data_frame = pandas.read_csv("confuse_sentences.csv")

for row in data_frame.itertuples(index=False):
    text, type = row[0], row[1]
    query_vector = sentence_model.encode(text, convert_to_tensor=True)
    similarity, pos = torch.max(util.pytorch_cos_sim(documents[type-1], query_vector), dim=0)
    print(text,"\t", text_pool[type-1][pos])

print('DONE')