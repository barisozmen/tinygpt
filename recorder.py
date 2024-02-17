from collections import defaultdict


class Recorder:
    def __init__(self):
        self.records = defaultdict(list)

    def record(self, event):
        event = list(event)
        event_type, event_values = event[0], event[1:]
        self.records[event_type].append(event_values)