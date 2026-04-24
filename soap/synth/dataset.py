import torch
from .synthesizer import synthesize_mixture

class SynthesizedDataSet:

    def __init__(
        self, size:int, channels:int, batch_size:int, length:int, alpha:list[float]=[1.0, 1.0, 2.0],
        device:torch.device=torch.device('cpu')
    ):
        self.alpha = alpha
        self.size = size
        self.channels = channels
        self.batch_size = batch_size
        self.num_batches = length // batch_size
        self.length = length
        self.device = device

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self._batch_idx = 0
        return self

    def __next__(self):
        if self._batch_idx >= self.num_batches:
            raise StopIteration
        self._batch_idx += 1
        return synthesize_mixture(
            self.size, self.channels, self.batch_size, self.device, self.alpha
        )

