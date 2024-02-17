import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import cycle, islice

from config import the_config

config = the_config()

class TextEncodingDataset(Dataset):
    def __init__(self, text, enc, verbose=True):
        self.data = torch.tensor(enc.encode(text), dtype=torch.long)
        i = int(len(self.data) * config.train_to_test_ratio)
        self.train = self.data[:i]
        self.validation = self.data[i:]
        if verbose:
            print('\n\n'+'-'*70)
            print(f"Train Dataset: {len(self.train)} tokens")
            print()
            print(f"example tokens:           {(example_tokens:=list(self.train[200:210].numpy()))}")
            print(f"example tokens (decoded): {[enc.decode([tok]) for tok in example_tokens]}")
            print()
            print(f'Encoder-> vocab_size:{enc.vocab_size}, name:{enc.name}')
            print('-'*70)

    def __len__(self):
        return (len(self.data) - config.block_size)

    def __getitem__(self, idx):
        block = self.data[idx:idx+config.block_size]
        x=block[:-1]
        y=block[1:]
        return x, y

# TOKENIZER
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    