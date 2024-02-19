from typing import Tuple
import time
import contextlib
from dataclasses import dataclass

import tiktoken
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.autograd.profiler import profile as torch_profile

from dataset import TextEncodingDataset
from encoder import make_encoder
from bigram_language_model import BigramLanguageModel

from config import the_config

from batch import Batch
from reporter import TrainingReporter



config = the_config() # Singleton class to store all hyperparameters and global variables

config.set(
    batch_size = 16, # number of independent sequences to train on in parallel
    block_size = 512, # max content lenght for predictions
    max_iters = 10000000,
    eval_interval = 100,
    train_to_test_ratio = 0.95,
    learning_rate = 1e-3,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    eval_iters = 100, # number of iterations to estimate the mean loss
    n_embd = 512, # number of embedding dimension
    n_head = 8, # number of heads in the multiheadattention models
    n_layer = 8, # number of sub-encoder-layers in the encoder
    dropout = 0.1, # dropout probability
    encoder_type = 'gpt2',
    token_freq_threshold = 4, # minimum frequency of a token to be included in the vocabulary
    torch_seed = 42, # random seed for reproducibility
)

profiler_config = {'use_cuda': True, 'with_flops': True, 'profile_memory': True}

with open('tiny_jules_verne.txt', 'r', encoding='utf-8') as f:
    text = f.read()

encoder = make_encoder(text)
dataset  = TextEncodingDataset(text, encoder)


model = BigramLanguageModel(encoder).to(config.device) # instantiate the model and move it to the gpu if available
model.print_model_description()

optimizer = torch.optim.AdamW(model.parameters() , lr=config.learning_rate) # nn.Model.parameters() returns a generator over all model parameters (torch.nn.parameter.Parameter), allowing the optimizer to access and update them.

batch = Batch(dataset, encoder)
batches = batch.generator()

training_reporter = TrainingReporter(model, batch)


for _ in range(config.max_iters):
    with training_reporter:
        x, y = next(batches)
        logits, loss = model(x, y)
        logits: Tensor; loss: Tensor
        optimizer.zero_grad(set_to_none=True) # clears out the old gradients
        loss.backward() # calculates new gradients and stores in ``.grad`` attribute of each parameter (torch.nn.parameter.Parameter), ready for the optimizer to use.
        optimizer.step() # updates the parameters using the .grad attribute of each parameter. remember that we instantiated the optimizer with the model parameters, so it knows which parameters to update.





model.dream_text(verbose=True, max_new_tokens=10000)

pass


# GPT Under the hood