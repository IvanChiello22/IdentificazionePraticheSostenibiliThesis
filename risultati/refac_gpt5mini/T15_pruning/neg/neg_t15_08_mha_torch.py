# 8) MultiheadAttention — T15 NEGATIVO: nessuna rimozione di head
import io
import torch, torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
x = torch.randn(3, 5, 32)

def estimate_head_energy(attn_module: nn.MultiheadAttention):
    # Simple per-head energy proxy based on head dimension
    embed_dim = attn_module.embed_dim
    num_heads = attn_module.num_heads
    head_dim = embed_dim // num_heads
    return [head_dim for _ in range(num_heads)]

energies = estimate_head_energy(mha)
# Energy-aware pruning decision: currently no heads selected to prune to preserve semantics
heads_to_prune = []

# Checkpointing (in-memory) to avoid knowledge loss in case of interruption
buf = io.BytesIO()
torch.save(mha.state_dict(), buf)
buf.seek(0)
mha.load_state_dict(torch.load(buf))

y,_ = mha(x, x, x)
print(y.shape)