# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/3/30 11:32 AM
==================================="""
from typing import List

from sentence_transformers import SentenceTransformer, util

import os
import torch


class SentenceModel:
    """
    Say something about the ExampleCalass...
    Args:
        args_0 (`type`):
        ...
    """

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"{model_path} not exists, please download model from "
                  f"https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.model = SentenceTransformer(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')

    def sentence_encode(self, data):
        embedding = self.model.encode(data, convert_to_tensor=True)
        # [1, 768]
        sentence_embeddings = util.normalize_embeddings(embedding)
        # embeddings shape: torch.Size([3, 768])
        print(f"embeddings shape: {sentence_embeddings.shape}")#
        return sentence_embeddings.tolist()


if __name__ == '__main__':

    senten_model = SentenceModel(model_path='/8t/workspace/lchang/models/paraphrase-multilingual-mpnet-base-v2')
    text2 = ["你好哇", "你好", "你的名字"]
    embeddings = senten_model.sentence_encode(text2)

    print(f"similarly score: {util.cos_sim(embeddings[0], embeddings[1:])}")
