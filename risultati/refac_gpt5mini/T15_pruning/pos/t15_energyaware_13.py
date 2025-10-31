import torch, torch.nn as nn, torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.nhead = nhead
        self.dk = d_model//nhead
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.o = nn.Linear(d_model, d_model)
    def forward(self, x):
        B,T,D = x.shape
        qkv = self.qkv(x)  # [B,T,3D]
        q,k,v = qkv.chunk(3, dim=-1)
        q = q.view(B,T,self.nhead,self.dk).transpose(1,2)  # [B,H,T,dk]
        k = k.view(B,T,self.nhead,self.dk).transpose(1,2)
        v = v.view(B,T,self.nhead,self.dk).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) / (self.dk**0.5)
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).contiguous().view(B,T,-1)
        return self.o(out), (q,k,v,attn)

def energy_per_head(T, dk, dv=None):
    dv = dk if dv is None else dv
    return T*T*dk + T*T + T*dv

def prune_heads_energy(module: MHA, keep_ratio=0.5, T=64):
    # checkpoint before destructive changes
    torch.save(module.state_dict(), 'mha_preprune.pth')
    with torch.no_grad():
        H = module.nhead; dk = module.dk
        device = module.o.weight.device
        energy = torch.tensor([energy_per_head(T, dk)]*H, dtype=torch.float32, device=device)
        # importance proxy: L1 dei pesi per head nella proiezione output (o in qkv)
        W_out = module.o.weight.detach()  # [D, D]
        per_head = W_out.view(module.nhead, dk, -1).abs().mean(dim=(1,2))
        score = per_head / (energy + 1e-9)
        k = max(1, int(keep_ratio*H))
        keep = torch.topk(score, k).indices.tolist()
        # Rimappa parametri riducendo teste (costruendo nuovi moduli Linear in modo sicuro)
        D = dk * H
        # build indices to keep across the 3 parts of qkv
        idx_keep = []
        for part in range(3):
            base = part * D
            for h in keep:
                idx_keep.extend(range(base + h*dk, base + (h+1)*dk))
        idx_keep_tensor = torch.tensor(idx_keep, dtype=torch.long, device=device)

        # Replace qkv Linear with trimmed version
        W_qkv = module.qkv.weight.data
        b_qkv = module.qkv.bias.data if module.qkv.bias is not None else None
        new_qkv_out = len(idx_keep)
        new_qkv = nn.Linear(module.qkv.in_features, new_qkv_out, bias=(b_qkv is not None)).to(device)
        new_qkv.weight.data.copy_(W_qkv[idx_keep_tensor, :].clone())
        if b_qkv is not None:
            new_qkv.bias.data.copy_(b_qkv[idx_keep_tensor].clone())
        module.qkv = new_qkv

        # Replace output projection to accept reduced input dim
        W_o = module.o.weight.data  # [Dout, Din_old]
        Dout = W_o.size(0)
        Dnew = dk * len(keep)
        new_o = nn.Linear(Dnew, Dout, bias=(module.o.bias is not None)).to(device)
        new_o.weight.data.copy_(W_o[:, :Dnew].clone())
        if module.o.bias is not None:
            new_o.bias.data.copy_(module.o.bias.data.clone())
        module.o = new_o

        module.nhead = len(keep)

mha = MHA(64, 4)
prune_heads_energy(mha, keep_ratio=0.5, T=64)
x = torch.randn(2,64,64)
y,_ = mha(x)
print(y.shape, 'heads=', mha.nhead)