import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class M(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.criterion = nn.CrossEntropyLoss()
        self._apply_energy_aware_pruning_from_env()

    def _apply_energy_aware_pruning_from_env(self):
        try:
            amount = float(os.getenv('PRUNE_RATE', '0.0'))
        except Exception:
            amount = 0.0
        if amount <= 0.0:
            return
        linear_modules = [(i, m) for i, m in enumerate(self.net) if isinstance(m, nn.Linear)]
        if not linear_modules:
            return
        _, most_expensive = max(linear_modules, key=lambda im: im[1].weight.numel())
        prune.l1_unstructured(most_expensive, name='weight', amount=amount)
        prune.remove(most_expensive, 'weight')

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-2)

class DS(torch.utils.data.Dataset):
    def __len__(self):
        return 256
    def __getitem__(self, i):
        return torch.randn(8), torch.randint(0, 2, (1,)).item()

ckpt = ModelCheckpoint(dirpath='ckpt_pl', save_top_k=1, monitor='train_loss', mode='min', filename='best')
pl.Trainer(max_epochs=3, callbacks=[ckpt], logger=False, enable_model_summary=False).fit(M(), torch.utils.data.DataLoader(DS(), batch_size=32))
print('done')