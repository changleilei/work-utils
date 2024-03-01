# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/2/21 7:20 PM
==================================="""
import torch


with open('data/input.txt', 'r', encoding='utf8') as f:
    text = f.read()
    # print(len(text))  # 1115394

    chars = sorted(list(set(text)))  # vacabulary
    # print(len(chars))  # 65

    print(chars)

    vocab_size = len(chars)
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    # print(char_to_idx)
    # print(idx_to_char)

def encode(text_):
    return [char_to_idx[char] for char in text_]

def decode(ids):
    return ''.join([idx_to_char[idx] for idx in ids])

# print(ii := encode('hii hello'))
# print(decode(ii))


def prepare_data(text, seq_len=32, batch_size=32):
    datas = torch.tensor(encode(text), dtype=torch.long)
    # print(datas.shape)  # torch.Size([1115394])
    # print(datas[:10])  # tensor([18, 47, 56, 57, 58,  1, 15, 47, 58, 47])

    n = int(0.9 * len(datas))
    train_data = datas[:n]
    val_data = datas[n:]

    # check datas
    # block_size = seq_len + 1
    # x = train_data[:block_size]
    # y = train_data[1:block_size + 1]
    #
    # for t in range(block_size):
    #     context = x[:t + 1]
    #     target = y[t]
    #     print(f"when index is {t}, context is {context}, target is {target}")
    return train_data, val_data

# prepare_data(text)

torch.manual_seed(1337)

batch_size = 4
block_size = 8

def get_batch(split):
    """
    获取批次数据
    :param split:
    :return:
    """
    train_data, val_data = prepare_data(text=text, seq_len=8, batch_size=batch_size)
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))  # (low, high, (shape))
    x = torch.stack([data[i: i+ block_size] for i in ix])
    y = torch.stack([data[i+1: i + block_size +1] for i in ix])
    return x, y


xb, yb = get_batch('train')
print('inputs')
print(xb.shape)
print(xb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")

