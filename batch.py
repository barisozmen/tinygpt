from itertools import cycle, islice
import torch

from config import Config

config = Config()

class Batch:
    def __init__(self, train_loader, val_loader, model, encoder):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.encoder = encoder

    def generator(self):
        for iter in range(1, config.max_iters+1):
            if iter % config.eval_interval == 0 or iter == config.max_iters:
                self.estimate_train_and_val_loss(iter, verbose=True)
                self.model.dream_text(max_new_tokens=200, verbose=True)

            x, y = self.get_random_batch(self.train_loader)
            yield x, y

    def get_random_batch(self, split='train'):
        data_loader = self.train_loader if split=='train' else self.val_loader
        ix = torch.randint(len(data_loader) - config.block_size, (config.batch_size,))
        x = torch.stack([data_loader.dataset[i:i+config.block_size] for i in ix])
        y = torch.stack([data_loader.dataset[i+1:i+config.block_size+1] for i in ix])
        x, y = x.to(config.device), y.to(config.device)
        return x, y
    
    def estimate_train_and_val_loss(self, iter, verbose=True):
        self.model.eval()
        train_loss = self.get_mean_loss(self.train_loader)
        val_loss = self.get_mean_loss(self.val_loader)
        self.model.train()
        if verbose:
            print('\n\n\n#####################################################')
            print(f"step {iter}: train loss {train_loss}, val loss {self.get_mean_loss(val_loss):.4f}")
            print('###################################################')
        return train_loss, val_loss
    
    @torch.no_grad() # tells pytorch to disable gradient calculation, making code run faster and with less memory. good practice to use when you are not going to backpropagate
    def get_mean_loss(self, loader):
        losses = torch.zeros(config.eval_iters)
        for iter in range(config.eval_iters):
            x, y = self.get_random_batch(loader)
            logits, loss = self.model(x.to(config.device), y.to(config.device))
            losses[iter] = loss.item()
        return losses.mean()