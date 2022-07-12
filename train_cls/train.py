# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/7/8 3:07 下午
==================================="""
import json

import pandas as pd
import tokenizer as tokenizer
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (TrainingArguments,
                          EvalPrediction,
                          AutoModelForSequenceClassification,
                          BertForSequenceClassification,
                          Trainer,
                          BertConfig,
                          AdamW)
from torch.utils.data import Dataset
import os
import numpy as np
from transformers import AutoTokenizer
from argparse import ArgumentParser
from datasets import load_dataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class EmotionDataset(Dataset):
    def __init__(self, data_path, model_path):
        super().__init__()
        self.data_path = data_path
        self.model_path = model_path
        self.labels = ['neutral', 'anger', 'happy', 'sadness']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_id = []
        self.load_data(self.data_path)

    def load_data(self, filename):
        # 加载数据
        print('loading data from:', filename)
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()[1:]
        for line in tqdm(lines, ncols=100):
            label, text = line.strip().split('\t')
            label_id = int(label)
            token = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
            self.input_ids.append(np.array(token['input_ids']))
            self.token_type_ids.append(np.array(token['token_type_ids']))
            self.attention_mask.append(np.array(token['attention_mask']))
            self.label_id.append(label_id)

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_id[index]

    def __len__(self):
        return len(self.input_ids)




def data_trans(data_path):
    d = []
    frame = pd.read_csv(data_path, sep='\t')
    for label, text in zip(frame['label'], frame['text_a']):
        temp_dic = {'label': label, 'text': text}
        d.append(temp_dic)

    return d


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier

from tqdm import tqdm


def main(args):
    # 参数设置
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    learning_rate = 5e-6  # Learning Rate不宜太大

    # 获取到dataset
    train_dataset = EmotionDataset(args.train_path, args.model_path)
    valid_dataset = EmotionDataset(args.valid_path, args.model_path)
    test_dataset = EmotionDataset(args.test_path, args.model_path)

    # 生成Batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(args.model_path)
    num_labels = len(train_dataset.labels)

    # 初始化模型
    model = BertClassifier(bert_config, num_labels).to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(1, epochs + 1):
        losses = 0  # 损失
        accuracy = 0  # 准确率

        model.train()

        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            # 传入数据，调用model.forward()
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            # 计算loss
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
            accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)

        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

        # 验证
        model.eval()
        losses = 0  # 损失
        accuracy = 0  # 准确率
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)

            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
                is_train=False
            )

            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
            accuracy += acc
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(valid_dataloader)
        average_acc = accuracy / len(valid_dataloader)

        print('\tValid ACC:', average_acc, '\tLoss:', average_loss)

        if not os.path.exists('models'):
            os.makedirs('models')

        # 判断并保存验证集上表现最好的模型
        if average_acc > best_acc:
            best_acc = average_acc
            torch.save(model.state_dict(), './results/pytorch_model.pkl')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../data/CPED/train.tsv",
                        help="Path of the train dataset for dist dataset. ")
    parser.add_argument("--valid_path", type=str, default="../data/CPED/valid.tsv",
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--test_path", type=str, default="../data/CPED/test.tsv",
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--log_path", type=str, default="", help="Path for store log")
    parser.add_argument("--train_batch_size", type=int, default=8, help="batch size for train")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="batch size for valid")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of train epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="train device")
    parser.add_argument("--n_embed", type=int, default=768, help="embedding layer dim")
    parser.add_argument("--model_path", type=str, default='/data/models/chinese-roberta-wwm-ext-large', help="model path")
    parser.add_argument("--num_labels", type=int, default=4, help="task for emotion classifier")
    parser.add_argument("--scheduler", type=str, default='linear', help="scheduler")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warmup Steps")

    parser.add_argument("--trained_weights", type=str, default='', help="training weights")
    args = parser.parse_args()
    main(args)
