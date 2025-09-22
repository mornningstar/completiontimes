from typing import Protocol

class Metric(Protocol):
    def name(self):
        ...
    
    def compute(self, y_true, y_pred):
        ...