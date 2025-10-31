import torch, torch.nn as nn

class TinyDecoder(nn.Module):
    def __init__(self, d=32, nhead=4):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=d, nhead=nhead, dim_feedforward=64)
        self.dec = nn.TransformerDecoder(layer, num_layers=2)
        self.emb = nn.Embedding(100, d)

    def forward(self, tgt, mem):
        t = self.emb(tgt)
        return self.dec(t, mem)

    def save_checkpoint(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.load_state_dict(torch.load(filepath))

mem = torch.randn(5, 2, 32)  # src len, batch, dim
tgt = torch.randint(0, 100, (7, 2))  # tgt len, batch
print(TinyDecoder()(tgt, mem).shape)