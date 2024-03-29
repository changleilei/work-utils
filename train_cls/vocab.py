# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/7/13 11:21 上午
==================================="""

from typing import List, Dict

import jieba
import nltk
import pandas as pd


class Vocab:
    """Build vocab for files and write to a file.
    Attributes:
        language: 'zh' for Chinese and 'en' for English
        word_dict: word frequency dict, {<word>: frequency}
            e.g. {'我': 30}
    """

    def __init__(self, language='zh', word_dict: Dict[str, int] = None):
        """Initialize with language type and word_dict.
        Args:
            language: 'zh' or 'en'
        """
        self.language = language
        self.word_dict = word_dict if word_dict else {}

    def load_file_to_dict(
            self, filename: str, cols: List[int] = None) -> Dict[str, int]:
        """Load columns of a csv file to word_dict.
        Args:
            filename: a csv file with ',' as separator
            cols: column indexes to be added to vocab
        Returns:
            word_dict: {<word>: frequency}
        """
        data_frame = pd.read_csv(filename)
        if not cols:
            cols = range(data_frame.shape[1])
        for row in data_frame.itertuples(index=False):
            for i in cols:
                sentence = str(row[i])
                if self.language == 'zh':
                    words = jieba.lcut(sentence)
                else:  # 'en'
                    words = nltk.word_tokenize(sentence)
                for word in words:
                    self.word_dict[word] = self.word_dict.get(word, 0) + 1
        return self.word_dict

    def write2file(self,
                   filename: str = 'vocab.txt', fre: bool = False) -> None:
        """Write word_dict to file without file head.
        Each row contains one word with/without its frequency.
        Args:
            filename: usually a txt file
            fre: if True, write frequency for each word
        """
        with open(filename, 'w', encoding='utf-8') as file_out:
            for word in self.word_dict:
                file_out.write(word)
                if fre:
                    file_out.write(' ' + str(self.word_dict[word]))
                file_out.write('\n')


def build_vocab(file_in, file_out):
    """Build vocab.txt for SMP-CAIL2020-Argmine."""
    vocab = Vocab('zh')
    vocab.load_file_to_dict(file_in, list(range(1, 3)))
    vocab.write2file(file_out, False)


if __name__ == '__main__':
    build_vocab('data/train.csv', 'vocab.txt')