from typing import Tuple

import tiktoken
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from dataset import TextEncodingDataset, CharLevelEncoder
from bigram_language_model import BigramLanguageModel
from config import the_config
from batch import Batch


config = the_config() # Singleton class to store all hyperparameters and global variables


config.set(
    batch_size = 16, # number of independent sequences to train on in parallel
    block_size = 32, # max content lenght for predictions
    max_iters = 10000,
    eval_interval = 100,
    train_to_test_ratio = 0.9,
    learning_rate = 1e-3, 
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    eval_iters = 200, # number of iterations to estimate the mean loss
    n_embd = 64, # number of embedding dimension
    n_head = 4, # number of heads in the multiheadattention models
    n_layer = 4, # number of sub-encoder-layers in the encoder
    dropout = 0.0, # dropout probability
    encoder_type = 'char_level', 
    torch_seed = 42, # random seed for reproducibility
)

with open('tiny_jules_verne.txt', 'r', encoding='utf-8') as f:
    text = f.read()

encoder = CharLevelEncoder(text) if config.encoder_type == 'char_level' else tiktoken.get_encoding('gpt2')
dataset  = TextEncodingDataset(text, encoder)
train_loader = DataLoader(dataset.train, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(dataset.validation, batch_size=config.batch_size, shuffle=True)

model = BigramLanguageModel(encoder).to(config.device) # instantiate the model and move it to the gpu if available
model.print_model_description()

optimizer = torch.optim.AdamW(model.parameters() , lr=config.learning_rate) # nn.Model.parameters() returns a generator over all model parameters (torch.nn.parameter.Parameter), allowing the optimizer to access and update them.
batch = Batch(train_loader, val_loader, model, encoder)

for x, y in batch.generator():
    logits, loss = model(x, y)
    logits: Tensor; loss: Tensor
    optimizer.zero_grad(set_to_none=True) # clears out the old gradients
    loss.backward() # calculates new gradients and stores in ``.grad`` attribute of each parameter (torch.nn.parameter.Parameter), ready for the optimizer to use.
    optimizer.step() # updates the parameters using the .grad attribute of each parameter. remember that we instantiated the optimizer with the model parameters, so it knows which parameters to update.

model.dream_text(verbose=True)

pass


# GPT Under the hood