from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ForwardCompletedEvent:
    config: object
    def __post_init__(self):
        self.d = self.config.__dict__.copy()

    def __iter__(self):
        return iter(['forward_completed', self.config])
    

class Recorder:
    def __init__(self):
        self.records = []

    def forward_completed(self, config):
        self.records.append(ForwardCompletedEvent(config))

    def records(self):
        return self.records
    

_the_recorder = None

def the_recorder():
    global _the_recorder
    if _the_recorder is None:
        _the_recorder = Recorder()
    return _the_recorder