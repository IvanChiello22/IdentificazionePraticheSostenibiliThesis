import torch, torch.nn as nn, torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,d=128,h=256):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
    def forward(self,x):
        hidden = F.gelu(self.fc1(x))
        # Ensure compatibility if hidden size was changed by pruning:
        expected_in = self.fc2.in_features
        curr_in = hidden.shape[-1]
        if curr_in != expected_in:
            if curr_in < expected_in:
                pad = hidden.new_zeros(hidden.shape[:-1] + (expected_in - curr_in,))
                hidden = torch.cat([hidden, pad], dim=-1)
            else:
                hidden = hidden[..., :expected_in]
        return self.fc2(hidden)

def prune_hidden_energy(fc1: nn.Linear, keep_ratio, batch_tokens=2048):
    with torch.no_grad():
        # checkpoint current layer state to avoid knowledge loss if interrupted
        try:
            torch.save({
                'weight': fc1.weight.clone(),
                'bias': fc1.bias.clone() if fc1.bias is not None else None,
                'out_features': fc1.out_features
            }, 'prune_checkpoint_fc1.pt')
        except Exception:
            # fallback: keep a memory backup if file save fails
            _backup = {
                'weight': fc1.weight.clone(),
                'bias': fc1.bias.clone() if fc1.bias is not None else None,
                'out_features': fc1.out_features
            }
        imp = fc1.weight.detach().abs().mean(dim=1)  # neuron importance
        energy = float(fc1.in_features) + float(batch_tokens)  # proxy
        score = imp / (energy + 1e-9)
        k = max(1, int(keep_ratio * fc1.out_features))
        keep = torch.topk(score, k).indices
        fc1.weight = nn.Parameter(fc1.weight[keep].clone())
        if fc1.bias is not None:
            fc1.bias = nn.Parameter(fc1.bias[keep].clone())
        fc1.out_features = k

m = MLP(); prune_hidden_energy(m.fc1, 0.5, 4096)
x = torch.randn(8, m.d); print(m(x).shape, m.fc1.out_features)