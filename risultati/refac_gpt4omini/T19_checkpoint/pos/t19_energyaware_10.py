import torch
import torch.nn as nn
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

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.CrossEntropyLoss()(self(x), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

class DS(torch.utils.data.Dataset):
    def __len__(self):
        return 256

    def __getitem__(self, i):
        return torch.randn(8), torch.randint(0, 2, (1,)).item()

ckpt = ModelCheckpoint(dirpath='ckpt_pl', save_top_k=1, monitor='train_loss', mode='min', filename='best')
trainer = pl.Trainer(max_epochs=3, callbacks=[ckpt], logger=False, enable_model_summary=False)
trainer.fit(M(), torch.utils.data.DataLoader(DS(), batch_size=32))

print('done')