# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/5/27 1:45 下午
==================================="""
from secrets import choice

"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MultipleNegativesRankingLoss. Entailnments are poisitive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset

Usage:
python training_nli_v2.py

OR
python training_nli_v2.py pretrained_transformer_model_name
"""
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random
import pandas as pd


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# model_name = sys.argv[1] if len(sys.argv) > 1 else 'distilroberta-base'
model_name = '/data/nli/3580000'
train_batch_size = 128          #The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 128
num_epochs = 5

# Save path of the model
model_save_path = 'training_nli_v2_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='gpu:1')

#Check if dataset exsist. If not, download and extract  it
# nli_dataset_path = 'data/AllNLI.tsv.gz'
# sts_dataset_path = 'data/stsbenchmark.tsv.gz'
train_data_path = '/data/nli/qa_match_cn_data/qa_match_data/train.csv'
valid_data_path = '/data/nli/qa_match_cn_data/qa_match_data/valid.csv'
test_data_path = '/data/nli/qa_match_cn_data/qa_match_data/test.csv'

# if not os.path.exists(nli_dataset_path):
#     util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)
#
# if not os.path.exists(sts_dataset_path):
#     util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
#

# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

all_sentences = []

nli_dataset_paths = []
train_df = pd.read_csv(train_data_path)

def count_len_of_all_sentences(all_sentences):
    count = 0
    for sentence in all_sentences:
        if len(sentence) < max_seq_length:
            count += 1
    print("count_len_of_all_sentences: ", count/len(all_sentences))

# get all sentences
all_sentences.extend(train_df['question'].tolist())
all_sentences.extend(train_df['answer'].tolist())
all_sentences = list(set(all_sentences))
count_len_of_all_sentences(all_sentences)
random.seed(123)
print(f"all sentences: {len(all_sentences)}")
train_samples = []
for index, row in train_df.iterrows():
    contracted_sentences = choice(all_sentences)
    question = str(row['question'])
    answer = str(row['answer'])
    while contracted_sentences == question or contracted_sentences == answer:
        contracted_sentences = choice(all_sentences)
    train_samples.append(InputExample(texts=[question, answer, contracted_sentences]))
    train_samples.append(InputExample(texts=[answer, question, contracted_sentences]))
logging.info("Train samples: {}".format(len(train_samples)))



# Special data loader that avoid duplicates within a batch
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)


# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)




#Read STSbenchmark dataset and use it as development set
# logging.info("Read STSbenchmark dev dataset")
# dev_samples = []
# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         if row['split'] == 'dev':
#             score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
#             dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
logging.info("Read qamatch dev dataset")
dev_samples = []
dev_df = pd.read_csv(valid_data_path)[['question', 'answer', 'score']]
for index, row in dev_df.iterrows():
  dev_samples.append(InputExample(texts=[str(row['question']), str(row['answer'])], label=row['score']))
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(train_dataloader)*0.1),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False          #Set to True, if your GPU supports FP16 operations
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

# test_samples = []
# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         if row['split'] == 'test':
#             score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
#             test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

test_samples = []
test_df = pd.read_csv(test_data_path)[['question', 'answer', 'score']]
for index, row in test_df.iterrows():
  test_samples.append(InputExample(texts=[str(row['question']), str(row['answer'])], label=row['score']))

model = SentenceTransformer(model_save_path, device='gpu:1')
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='qa_match-test')
test_evaluator(model, output_path=model_save_path)