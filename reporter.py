
from abc import ABC, abstractmethod
import contextlib
from dataclasses import dataclass, field
import time
import re

import torch

from config import the_config
from batch import Batch

config = the_config()


class BaseReporter(ABC):
    def __init__(self, treporter):
        self.treporter = treporter

    def start(self): pass

    @abstractmethod
    def report(self): pass

    def should_report(self, iter):
        return iter % config.eval_interval == 0 or iter >= config.max_iters

class Timer(BaseReporter):
    def __init__(self, treporter):
        super().__init__(treporter)
        self.training_start_time = time.time()
    def start(self):
        self.batch_start_time = time.time()
    def report(self):
        print(f"""Time passed for
        this batch : {(t:=time.time()) - self.batch_start_time:.4f} sec
        training   : {t - self.training_start_time:.4f} sec
        """)

class LossEstimator(BaseReporter):
    def report(self):
        model = self.treporter.model
        batch = self.treporter.batch

        model.eval()
        train_loss = LossEstimator.get_mean_loss(model, batch.train_dict)
        val_loss = self.get_mean_loss(model, batch.validation_dict)
        model.train()

        print(f"""train loss : {train_loss:.4f}\nval loss   : {val_loss:.4f}""")
    
    @staticmethod
    @torch.no_grad() # tells pytorch to disable gradient calculation, making code run faster and with less memory. good practice to use when you are not going to backpropagate
    def get_mean_loss(model, data_dict):
        losses = torch.zeros(config.eval_iters)
        for iter in range(config.eval_iters):
            x, y = Batch.get_pseudo_random_batch(data_dict)
            logits, loss = model(x.to(config.device), y.to(config.device))
            losses[iter] = loss.item()
        return losses.mean()
    

class IterationReporter(BaseReporter):
    def report(self):
        print(f"\n\n\n{'#'*50}   iteration {self.treporter.iter}   {'#'*30}")


class Oneirocritic(BaseReporter):
    def report(self):
        dream = self.treporter.model.dream_text(max_new_tokens=200)
        re.sub(r'\n+', '\n', dream)
        print(f'\nModel Dreams:\n{"-"*100}\n' + dream + f'\n{"-"*100}')


@dataclass
class TrainingReporter(contextlib.ContextDecorator):
    model: torch.nn.Module
    batch: Batch
    iter: int = 0

    def __post_init__(self):
        self.reporters = [cls(self) for cls in (IterationReporter, Timer, LossEstimator, Oneirocritic)]

    def __enter__(self):
        self.iter += 1
        [r.start() for r in self.reporters]

    def __exit__(self, *exc):
        [r.report() for r in self.reporters if r.should_report(self.iter)]