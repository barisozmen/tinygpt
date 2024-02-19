from dataclasses import dataclass

from random import shuffle
import math
import numpy as np
from itertools import cycle, islice

import torch

from config import the_config

config = the_config()

@dataclass
class Batch:
    dataset: object
    encoder: object
    def __post_init__(self):
        self.train_dict = self._make_data_dict(self.dataset.train)
        self.validation_dict = self._make_data_dict(self.dataset.validation)

    def generator(self):
        while True:
            x, y = Batch.get_pseudo_random_batch(self.train_dict)
            yield x, y
        
    @staticmethod
    def get_pseudo_random_batch(data_dict):
        data, ix_generator = data_dict['data'], data_dict['ix_generator']
        ix = next(ix_generator)
        x = torch.stack([data[i:i+config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
        x, y = x.to(config.device), y.to(config.device)
        return x, y
    
    def _make_data_dict(self, data):
        ix = list(range(len(data)-config.block_size))
        shuffle(ix)
        return {
            'data': data,
            'tokens': len(data),
            'ix_generator': cycle(split_list(ix, chunk_size=config.batch_size)),
        }
    

def split_list(lst, chunk_size):
    return np.array_split(lst, math.ceil(len(lst)/chunk_size))[:-1]