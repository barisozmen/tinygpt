from collections import Counter
from abc import ABC, abstractmethod
import random

import tiktoken
import torch

from config import the_config

config = the_config()

class Encoder(ABC):
    def __init__(self, text):
        self.tokens: list = self._get_tokens(text)
        self._make(self.tokens)

    def _make(self, tokens):
        tokens = sorted(tokens)
        self.n_vocab = len(tokens)
        self.stoi = { ch:i for i, ch in enumerate(tokens) }
        self.itos = { i:ch for i, ch in enumerate(tokens) }
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])

    @abstractmethod
    def _get_tokens(self, text): pass
        

class CharLevelEncoder(Encoder):
    def _get_tokens(self, text):
        return list(set(text)) # all the unique characters


# class MinimalGPT2Encoder(Encoder):
#     def _get_tokens(self, text):
#         return self._remove_low_frequency_tokens(self._get_token_frequencies(text))

#     def _get_token_frequencies(self, text):
#         enc = tiktoken.get_encoding('gpt2')
#         encoded_data = torch.tensor(enc.encode(text), dtype=torch.long)
#         return list(map(lambda x: (enc.decode([x[0]]), x[1]), Counter(encoded_data.numpy()).most_common()))
    
#     def _remove_low_frequency_tokens(self, freqs):
#         return list(map(lambda x: x[0], filter(lambda x: x[1] >= config.token_freq_threshold, freqs)))

def get_token_frequencies(enc, text):
    encoded_data = torch.tensor(enc.encode(text), dtype=torch.long)
    return Counter(encoded_data.numpy()).most_common()

def remove_low_frequency_tokens(freqs):
    return list(map(lambda x: x[0], filter(lambda x: x[1] >= config.token_freq_threshold, freqs)))

def minimizer(enc, text): 
    minimized_token_integers = set(remove_low_frequency_tokens(get_token_frequencies(enc, text)))
    original_encode = enc.encode
    original_decode = enc.decode

    def new_encode(text):
        res = original_encode(text)
        return [k if k in minimized_token_integers else random.choice(list(minimized_token_integers)) for k in res]

    def new_decode(nums):
        nums = [k if k in minimized_token_integers else random.choice(list(minimized_token_integers)) for k in nums]
        return original_decode(nums)

    enc.encode = new_encode
    enc.decode = new_decode
    enc.vocab_size = len(minimized_token_integers)

    return enc

def make_encoder(text):
    match config.encoder_type:
        case 'char_level':
            enc = CharLevelEncoder(text)
        case 'gpt2':
            enc = tiktoken.get_encoding('gpt2')
        case 'minimal_gpt2' | _:
            enc = minimizer(tiktoken.get_encoding('gpt2'), text)

    for attr in ['n_vocab', 'encode', 'decode']:
        assert hasattr(enc, attr)

    return enc