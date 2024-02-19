import torch
from torch import nn
import torch.nn.functional as F

from config import the_config
from language_model import LanguageModel

"""
Inspired from Andrej Karpathy's famous video "Let's build GPT: from scratch, in code, spelled out." - https://www.youtube.com/watch?v=kCc8FmEb1nY

Below are the pytorch.nn modules used in this code and their explanations:
- nn.Module --> Base class for all neural network modules. Your models should also subclass this class.
- nn.Linear --> Applies a linear transformation to the incoming data: y = xA^T + b. Used for the key, query, and value transformations in the SelfAttentionHead. Also used in the feedforward layers.
- nn.ModuleList -->  Holds a list of modules. Used to hold the multiple SelfAttentionHead instances in the MultiHeadSelfAttention module.
- nn.Dropout -->  Randomly zeroes some of the elements of the input tensor with probability p. Used for regularization and preventing overfitting.
- nn.LayerNorm -->  Applies layer normalization over a mini-batch of inputs. Helps stabilize learning. Used before self-attention and feedforward layers.
- nn.Sequential -->  Chains a sequence of modules together. Used to chain the feedforward neural network layers.
- nn.Embedding -->  Maps integers (vocabulary indices) to vectors of a fixed dimension. Used to lookup token and position embeddings.
- nn.Linear -->  Maps input to output in a linear fashion. Used in the final language modeling head to map to vocabulary size.
- F.cross_entropy -->  Applies cross entropy loss between logits and targets. Used to calculate language modeling loss.
- F.softmax -->  Applies softmax over given dimension. Used to convert logits to probabilities.
- torch.multinomial -->  Samples from a multinomial distribution. Used to sample the next token from the network's predicted token probabilities.


Below are papers and resources that explain the concepts used in this code:
- "Attention is All You Need" (https://arxiv.org/abs/1706.03762) -> main paper for the transformer architecture
- "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385) -> residual connections
- "Layer Normalization" (https://arxiv.org/abs/1607.06450)
- "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

This is a decoder-only transformer model. It lacks encoder and cross-attention modules from the original paper - attention is all you need. Karpathy explains what is left out and why at https://youtu.be/kCc8FmEb1nY?si=pBIqxDI3jxZyEx0I&t=6160. 
- The reason the original paper has an encoder-decoder architecture is that it is designed for machine translation tasks. The decoder-only architecture is used for language modeling tasks.
- The encoder makes the model pre-conditioned on a given data (text for this case), such as in machine translation
- The original paper uses cross-attention to allow the decoder to attend to the encoder's output. This is useful for tasks like machine translation, where the decoder needs to attend to the input sentence to generate the output sentence.

"""


config = the_config()

class SelfAttentionHead(nn.Module):
    """ one head of self-attention

    Notation:
    [B]atch -> batch size
    [T]ime -> sequence length
    [C]hannel -> embedding size, or vocab size

    Self-attention also called KQV-attention K: key, Q: query, V: value

    By the "Attention all you need" paper: Attention(K,Q,V) = softmax( KQ^T / sqrt(C) ) V
    """

    def __init__(self, head_size):
        super().__init__()
        # Every element (corresponding to a token) in the input sequence emits three vectors: 
        # - key (what do I contain)
        # - query (what am I looking for)
        # - and value (what do I communicate to others).

        # nn.Linear applies a linear transformation to the incoming data: y = xA^T + b
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)

        # Register a lower triangular matrix of 1's as buffer. It will be used to mask future positions in the input sequence,
        # ensuring that a position can only attend to previous positions and itself. This is crucial for autoregressive models 
        # to prevent information leakage from future tokens.
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)
        
        # compute attention scores ("affinities") - the dot product of the query and key vectors
        weights = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T). i.e Weights are scaled by T^2 for each element in the batch
        
        # Karpathy explains the math trick here at https://youtu.be/kCc8FmEb1nY?si=l61egLEzrfTdIKss&t=3391
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) 
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        
        # perform the weighted aggregation of the values
        out = weights @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadSelfAttention(nn.Module):
    """ multiple self-attention heads in parallel 
    
    Holds a list of SelfAttentionHead instances and projects the concatenated outputs to the desired dimension.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.self_attention_heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.self_attention_heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class PositionwiseFeedForward(nn.Module):
    """Implements a two-layer feed-forward network."""
    def __init__(self, n_embd, ffwd_expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, ffwd_expansion * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadSelfAttention(n_head, head_size)
        self.feed_forward = PositionwiseFeedForward(n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd) # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Apply self-attention and add residual connection (aka skip connection). 
        # Adding residual connections idea comes from the "Deep Residual Learning for Image Recognition" paper (https://arxiv.org/abs/1512.03385). It helps to train very deep networks, make them easier to optimize.
        x = x + self.attention(self.layer_norm_1(x)) # In the original "Attention All You Need" paper, the layer normalization is applied after the self attention. With the advences in the field (by 2024 January), it is now common to apply layer normalization before the self attention.
        x = x + self.feed_forward(self.layer_norm_2(x)) # same above
        return x

class BigramLanguageModel(LanguageModel):
    def __init__(self, *argrs, **kwargs):
        super().__init__(*argrs, **kwargs)
        # Each token looks up the logits for the next token from token_embedding_table
        # nn.Embedding layer maps integers (vocab indices) to vectors of a fixed dimension
        self.token_embedding_table = nn.Embedding(self.encoder.vocab_size, config.n_embd)

        # Position embeddings are learned representations for position in the sequence
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        # Blocks implement the transformer architecture in batches
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config.n_embd, n_head=config.n_head) for _ in range(config.n_layer)])

        # Final layer norm 
        self.final_norm = nn.LayerNorm(config.n_embd)

        # Language modeling head. Maps last transformer block output to vocab size
        self.lm_head = nn.Linear(config.n_embd, self.encoder.vocab_size)

    def _forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.transformer_blocks(x) # (B,T,C)
        x = self.final_norm(x) # (B,T,C)
        # Project to vocab size to get next-token logits
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # Reshape the logits tensor from shape (B, T, C) to shape (B*T, C)
            targets = targets.view(B*T) # Reshape the targets tensor from shape (B, T) to shape (B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
