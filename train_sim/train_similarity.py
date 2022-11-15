# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/6/7 4:46 下午
==================================="""
import math
from datetime import datetime
from sentence_transformers import SentenceTransformer, InputExample, losses,datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


model_name = './chinese-roberta-wwm-ext'
model_save_path = 'robert-similar' + model_name.replace("/", "-") + '-' + datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")
train_data_path = '/data/nli/sts_datas/train1.txt'
valid_data_path = '/data/nli/sts_datas/valid1.txt'
test_data_path = '/data/nli/sts_datas/test1.txt'

num_epochs = 20
train_batch_size = 64
device = 'cuda:3'

def get_txt(data_path):
    with open(data_path, 'r', encoding='utf8') as f:
        return f.readlines()

def get_inputs(data_rows):
    samples = []
    for row in data_rows:
        texts = row.strip().split('\t')
        if len(texts) != 3:
            print(row)
            continue
        samples.append(InputExample(texts=texts[:2], label=float(texts[2])))

    return samples


# 获取训练数据
train_data_rows = get_txt(train_data_path)
train_samples = get_inputs(train_data_rows)


# 获取验证数据
dev_data_rows = get_txt(valid_data_path)
dev_samples = get_inputs(dev_data_rows)

# 模型构建
model = SentenceTransformer(model_name, device=device)
loss = losses.CosineSimilarityLoss(model)


train_dataset_loader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
warmup_steps = math.ceil(len(train_dataset_loader) * num_epochs * 0.1)  # 10% of train data for warm-up

print("Warmup-steps: {}".format(warmup_steps))


dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                 name='cn-sts-dev')

# train model

model.fit(train_objectives=[(train_dataset_loader, loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(train_dataset_loader) * 0.2),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False
          )

test_data_rows = get_txt(test_data_path)
test_data_samples = get_inputs(test_data_rows)

model = SentenceTransformer(model_save_path, device=device)

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_data_samples, batch_size=train_batch_size,
                                                                  name='cn-sts-test')
test_evaluator(model, output_path=model_save_path)