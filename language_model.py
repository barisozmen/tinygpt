import torch

from config import Config

config = Config()

# nn.Module is base class for all pytorch neural network modules
class LanguageModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def print_model_description(self):
        print('\n'); print(self); print('\n')
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())/1e6} Million")

    def dream_text(self, max_new_tokens=2000, verbose=True):
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
        dream = self.encoder.decode(self.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
        if verbose:
            print(dream)
        return dream