import torch, torch.nn as nn, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.utils.prune as prune

class M(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,2))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-2)

    def _energy_rank_linear_layers(self):
        ranks = []
        for name, module in self.net.named_modules():
            if isinstance(module, nn.Linear):
                params = sum(p.numel() for p in module.parameters())
                ranks.append((name, module, params))
        ranks.sort(key=lambda x: -x[2])  # most parameters (proxy for energy) first
        return ranks

    def prune_most_energy_intensive(self, amount=0.0):
        if amount <= 0.0:
            return
        ranks = self._energy_rank_linear_layers()
        if not ranks:
            return
        # prune the most "energy intensive" (most params) linear layer
        _, module, _ = ranks[0]
        prune.l1_unstructured(module, name='weight', amount=amount)

class DS(torch.utils.data.Dataset):
    def __len__(self): return 128
    def __getitem__(self, i): return torch.randn(8), torch.randint(0,2,(1,)).item()

checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', save_last=True)
trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=True, enable_model_summary=False, callbacks=[checkpoint_callback])

model = M()
# energy-aware pruning hook present; default amount=0.0 to preserve behavior
model.prune_most_energy_intensive(amount=0.0)

trainer.fit(model, torch.utils.data.DataLoader(DS(), batch_size=32))
print('done')