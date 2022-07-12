# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/7/11 3:40 下午
==================================="""
import torch
import torch.nn as nn
from transformers import BertModel


# Bert
class BertClassifier(nn.Module):
    def __init__(self, bert_config, num_labels):
        super().__init__()
        # 定义BERT模型
        self.bert = BertModel(config=bert_config)
        # dropout
        self.dropout = nn.Dropout(0.5)
        # 打开BERT训练
        for param in self.bert.parameters():
            param.requires_grad = True
        # 定义分类器
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, is_train=True):
        # BERT的输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取[CLS]位置的pooled output
        pooled = bert_output[1]
        if is_train:
            pooled = self.dropout(pooled)
        # 分类
        logits = self.classifier(pooled)
        # 返回softmax后结果
        return torch.softmax(logits, dim=1)


