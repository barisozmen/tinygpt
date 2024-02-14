import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import cycle, islice

from config import Config


config = Config()


class CharLevelEncoder:
    def __init__(self, text):
        self.chars = sorted(list(set(text))) # all the unique characters
        self.n_vocab = len(self.chars)
        self.stoi = { ch:i for i, ch in enumerate(self.chars) }
        self.itos = { i:ch for i, ch in enumerate(self.chars) }
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])


class TextEncodingDataset(Dataset):
    def __init__(self, text, enc):
        self.data = torch.tensor(enc.encode(text), dtype=torch.long)
        i = int(len(self.data) * config.train_to_test_ratio)
        self.train = self.data[:i]
        self.validation = self.data[i:]

    def __len__(self):
        return (len(self.data) - config.block_size)

    def __getitem__(self, idx):
        block = self.data[idx:idx+config.block_size]
        x=block[:-1]
        y=block[1:]
        return x, y

# TOKENIZER
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb