# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/5/27 9:54 上午
==================================="""
import math
import random


from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import pandas as pd
import numpy as np

from random import choice
from sentence_transformers.cross_encoder import CrossEncoder

train_data_path = '/data/nli/qa_match_cn_data/train.csv'
valid_data_path = '/data/nli/qa_match_data/valid.csv'
test_data_path = '/data/nli/qa_match_data/test.csv'
seed = 123
np.random.seed(seed)
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = '/data/nli/3580000'
train_batch_size = 32  # The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 128
num_epochs = 10

# Save path of the model
model_save_path = '/data/nli/stsb-robert-qa-match-zh' + '-' + datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S")



# Check if dataset exsist. If not, download and extract  it
# nli_dataset_dir = '/content/drive/MyDrive/qa_match_data/train.csv'
#sts_dataset_path = 'data/stsbenchmark.tsv.gz'

#if not os.path.exists(sts_dataset_path):
#    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read RCTNLI train dataset")


def read_csv(path):
    df = pd.read_csv(path)
    return df

def count_len_of_all_sentences(all_sentences):
    count = 0
    for sentence in all_sentences:
        if len(sentence) < max_seq_length:
            count += 1
    print("count_len_of_all_sentences: ", count/len(all_sentences))

all_sentences = []
all_df = pd.DataFrame({})
nli_dataset_paths = ['/data/nli/qa_match_cn_data/train.csv']

#for _, _, filenames in os.walk(nli_dataset_dir):
#    for filename in filenames:
#        print(os.path.join(nli_dataset_dir, filename))
#        if os.path.splitext(filename)[-1] == '.csv':
#            nli_dataset_paths.append(os.path.join(nli_dataset_dir, filename))

for path in nli_dataset_paths:
    df = read_csv(path)
    df = df[df['score'] == 1]

    # get all sentence pairs
    if all_df.empty:
        all_df = df[['question', 'answer', 'score']]
        all_df = all_df.dropna(axis=0, how='any')
    else:
        tmp_df = df[['question', 'answer', 'score']]
        all_df = pd.concat([all_df, tmp_df])
        all_df = all_df.dropna(axis=0, how='any')

#all_df = all_df[all_df['answers'].str.len() > 20]
# get all sentences
all_sentences.extend(all_df['question'].tolist())
all_sentences.extend(all_df['answer'].tolist())
all_sentences = list(set(all_sentences))
count_len_of_all_sentences(all_sentences)
random.seed(123)
print(f"all sentences: {len(all_sentences)}")
train_samples = []
for index, row in all_df.iterrows():
    label = np.random.uniform(low=0.8, high=1.0)
    train_samples.append(InputExample(texts=[row['question'], row['answer']], label=row['score']))
    train_samples.append(InputExample(texts=[row['answer'], row['question']], label=1-row['score']))
    # train_samples.append(InputExample(texts=[row['questions'], row['answers'], choice(all_sentences)]))

# for sent1, others in train_data.items():
#     if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
#         train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
# train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))


# Here we define our SentenceTransformer model
# word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
from torch.utils.data import DataLoader
model = CrossEncoder('/content/drive/MyDrive/3580000', num_labels=1)
logging.info("Train samples: {}".format(len(train_samples)))
train_batch_size = 8
# Special data loader that avoid duplicates within a batch
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# Our training loss
# train_loss = losses.MultipleNegativesRankingLoss(model)


# Configure the training
warmup_steps = math.ceil(len(all_df) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

# Read STSbenchmark dataset and use it as development set
logging.info("Read qamatch dev dataset")
dev_samples = []
dev_df = read_csv(valid_data_path)[['question', 'answer', 'score']]
for index, row in dev_df.iterrows():
  dev_samples.append(InputExample(texts=[row['question'], row['answer']], label=row['score']))

evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples, name='qamatch-dev')
# test_evaluator(model)

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(all_df) * 0.1),
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################


# dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
#                                                                  name='qa-match-dev')

test_samples = []
test_df = read_csv(test_data_path)[['question', 'answer', 'score']]
for index, row in test_df.iterrows():
  test_samples.append(InputExample(texts=[row['question'], row['answer']], label=row['score']))

test_evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(test_samples, batch_size=train_batch_size,
                                                                  name='qa-match-test')

model = SentenceTransformer(model_save_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size,
#                                                                   name='sts-test')
test_evaluator(model, output_path=model_save_path)