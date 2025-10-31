import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.prune as prune

enc = nn.LSTM(10, 20, batch_first=True)
dec = nn.LSTM(20, 20, batch_first=True)

def apply_pruning(start_layer, next_layer, amount=0.0):
    def _prune_layer(layer, amt):
        if amt <= 0.0:
            return
        for name in ('weight_ih_l0', 'weight_hh_l0'):
            if hasattr(layer, name):
                prune.l1_unstructured(layer, name=name, amount=amt)
    # start from the (assumed) more energy-intensive layer
    _prune_layer(start_layer, amount)
    _prune_layer(next_layer, amount)

apply_pruning(dec, enc, amount=0.0)

x = torch.randn(2, 7, 10)
def _enc_forward(x):
    return enc(x)
h, _ = checkpoint(_enc_forward, x)
def _dec_forward(inp):
    return dec(inp)
y, _ = checkpoint(_dec_forward, h)
print(y.shape)