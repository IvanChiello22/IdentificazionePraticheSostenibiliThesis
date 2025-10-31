import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.prune as prune

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x)); x = F.adaptive_avg_pool2d(F.relu(self.c3(x)),1).view(x.size(0),-1)
        return self.fc(x)

def ho_wo(H,W,k,p,s,d):
    Ho = (H + 2*p - d*(k-1) - 1)//s + 1
    Wo = (W + 2*p - d*(k-1) - 1)//s + 1
    return Ho, Wo

def layer_energy(conv, in_shape):
    B,C,H,W = in_shape
    kh,kw = conv.kernel_size; s = conv.stride[0]; p = conv.padding[0]; d = conv.dilation[0]
    Ho,Wo = ho_wo(H,W,kh,p,s,d)
    macs = Ho*Wo*conv.out_channels*(conv.in_channels//conv.groups)*kh*kw
    act = Ho*Wo*conv.out_channels
    return 2*macs + act  # proxy energia

def energy_splits(layers, in_shapes, target_sparsity=0.4, keep_floor=0.2):
    E = torch.tensor([layer_energy(l, ish) for l,ish in zip(layers,in_shapes)], dtype=torch.float32)
    share = E / E.sum()
    # Al più prune = target_sparsity * share per layer
    keep = torch.clamp(1.0 - target_sparsity*share, keep_floor, 0.95).tolist()
    return keep

def prune_out(conv, keep_ratio, in_shape):
    with torch.no_grad():
        B,C,H,W = in_shape
        kh,kw = conv.kernel_size; Ho,Wo = ho_wo(H,W,kh,conv.padding[0],conv.stride[0],conv.dilation[0])
        energy = Ho*Wo*conv.in_channels*kh*kw
        Wabs = conv.weight.detach().abs()
        l1 = Wabs.view(Wabs.size(0), -1).mean(1)
        score = l1 / (energy + 1e-9)
        k = max(1, int(keep_ratio*conv.out_channels))
        if k >= conv.out_channels:
            return
        # Use built-in structured pruning to compute which filters to remove (by L1 across filter)
        amount = float(conv.out_channels - k) / conv.out_channels
        prune.ln_structured(conv, name='weight', amount=amount, n=1, dim=0)
        # extract mask and determine kept indices
        mask = getattr(conv, 'weight_mask', None)
        if mask is None:
            # fallback: compute topk by score
            keep = torch.topk(score, k).indices
        else:
            row_mask = mask.view(mask.size(0), -1).abs().sum(1) > 0
            keep = torch.nonzero(row_mask, as_tuple=False).view(-1)
        # materialize pruned weights by selecting kept filters and updating attributes
        effective_weight = conv.weight.detach().clone()
        new_weight = effective_weight[keep].clone()
        conv.weight = nn.Parameter(new_weight)
        if conv.bias is not None:
            eff_bias = conv.bias.detach().clone()
            conv.bias = nn.Parameter(eff_bias[keep].clone())
        conv.out_channels = new_weight.size(0)
        # Clean up any pruning reparameterization artifacts if present
        try:
            # Attempt safe removal of pruning reparam; this will set weight to current value if possible
            if hasattr(conv, 'weight_orig') or hasattr(conv, 'weight_mask'):
                # remove entries from internal dicts if present
                if 'weight_orig' in conv._parameters:
                    conv._parameters.pop('weight_orig')
                if 'weight_mask' in conv._buffers:
                    conv._buffers.pop('weight_mask')
        except Exception:
            pass
        # Ensure parameter registration is consistent
        conv._parameters['weight'] = conv.weight

net = Net()
layers = [net.c1, net.c2, net.c3]
in_shapes = [(1,3,64,64), (1,16,64,64), (1,32,32,32)]
keeps = energy_splits(layers, in_shapes, target_sparsity=0.5, keep_floor=0.3)
# checkpoint before pruning to avoid knowledge loss in case of interruption
torch.save(net.state_dict(), 'checkpoint_before_prune.pth')
for l, k, ish in zip(layers, keeps, in_shapes): prune_out(l, k, ish)
# checkpoint after pruning
torch.save(net.state_dict(), 'checkpoint_after_prune.pth')
x = torch.randn(2,3,64,64); print(net(x).shape, [l.out_channels for l in layers], [round(k,2) for k in keeps])