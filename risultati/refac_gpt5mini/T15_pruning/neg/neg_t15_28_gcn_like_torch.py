# 28) GCN-like — T15 NEGATIVO: nessun pruning di nodi
import torch, torch.nn as nn
A = torch.tensor([[1,1,0],[1,1,1],[0,1,1]], dtype=torch.float32)
X = torch.randn(3, 16)
W = nn.Parameter(torch.randn(16, 8))

def prune_parameter_magnitude(param: nn.Parameter, amount: float = 0.0) -> nn.Parameter:
    if amount <= 0.0:
        return param
    tensor = param.data.clone()
    num_to_prune = int(tensor.numel() * amount)
    if num_to_prune == 0:
        return param
    flat_abs = tensor.abs().view(-1)
    threshold = torch.topk(flat_abs, num_to_prune, largest=False)[0].max()
    mask = tensor.abs() > threshold
    pruned_tensor = tensor * mask
    return nn.Parameter(pruned_tensor)

def checkpoint_parameter(param: nn.Parameter, path: str = None) -> None:
    if path:
        torch.save(param.detach(), path)

# Sustainable tactics configuration (no-op by default to preserve original behavior)
_prune_amount = 0.0      # fraction of weights to prune (0.0 = no pruning)
_checkpoint_path = None # set to a filepath to enable checkpointing, e.g., 'w_checkpoint.pt'

# Apply checkpointing and pruning minimally
checkpoint_parameter(W, _checkpoint_path)
W_effective = prune_parameter_magnitude(W, _prune_amount)

Z = A @ X @ W_effective
print(Z.shape)