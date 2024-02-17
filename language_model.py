import torch
import torch.nn as nn

from config import the_config

config = the_config()

# nn.Module is base class for all pytorch neural network modules
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def print_model_description(self):
        print('\n'); print(self); print('\n')
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())/1e6} Million\n")
        print("Hyperparameters: ", vars(config), '\n')

class LanguageModel(MyModel):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def dream_text(self, max_new_tokens=2000, verbose=False):
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
        dream = self.encoder.decode(self.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
        if verbose: print(dream)
        return dream
    
    def print_model_description(self):
        super().print_model_description()
        print(f"[Text Encoder] name={self.encoder.name}, vocab_size={self.encoder.vocab_size}")
