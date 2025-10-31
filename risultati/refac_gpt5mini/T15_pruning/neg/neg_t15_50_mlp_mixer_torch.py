# 50) MLP-Mixer mini — T15 NEGATIVO: nessuna sparsificazione/pruning
import torch, torch.nn as nn
import torch.nn.utils.prune as prune

class Mixer(nn.Module):
    def __init__(self, tokens=16, dims=32):
        super().__init__()
        self.token_mlp = nn.Sequential(nn.Linear(tokens, tokens*2), nn.GELU(), nn.Linear(tokens*2, tokens))
        self.channel_mlp = nn.Sequential(nn.Linear(dims, dims*2), nn.GELU(), nn.Linear(dims*2, dims))
    def forward(self, x):  # x: B,T,D
        y = x + self.token_mlp(x.transpose(1,2)).transpose(1,2)
        z = y + self.channel_mlp(y)
        return z

def energy_aware_prune(model: nn.Module, amount: float = 0.2):
    """
    Prune linear layer weights starting from the largest (proxy for energy intensity)
    until a fraction `amount` of total linear weights is pruned.
    Uses L1 unstructured pruning (PyTorch official API).
    """
    if amount <= 0.0:
        return
    # collect linear weight modules and their sizes
    linear_weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
            linear_weights.append((module, module.weight.numel()))
    if not linear_weights:
        return
    total_params = sum(n for _, n in linear_weights)
    target_prune = amount * total_params
    # sort descending by number of params (largest -> most energy intensive)
    linear_weights.sort(key=lambda t: t[1], reverse=True)
    pruned_so_far = 0.0
    for module, nparams in linear_weights:
        remaining = target_prune - pruned_so_far
        if remaining <= 0:
            break
        # fraction of this layer to prune
        layer_amount = min(1.0, remaining / nparams)
        if layer_amount <= 0.0:
            continue
        prune.l1_unstructured(module, name='weight', amount=layer_amount)
        pruned_so_far += layer_amount * nparams

def save_checkpoint(model: nn.Module, optimizer=None, epoch: int = None, path: str = 'checkpoint.pth'):
    """
    Save model (and optionally optimizer) state_dict as a checkpoint.
    Minimal utility for training resilience.
    """
    ckpt = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        ckpt['epoch'] = epoch
    torch.save(ckpt, path)

def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    """
    Load checkpoint into model (and optionally optimizer). Returns loaded epoch if present.
    """
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt.get('model_state_dict', {}))
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', None)

x = torch.randn(2,16,32); print(Mixer()(x).shape)