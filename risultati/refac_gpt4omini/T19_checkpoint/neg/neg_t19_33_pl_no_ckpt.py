import torch, torch.nn as nn, pytorch_lightning as pl

class M(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss')

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.CrossEntropyLoss()(self(x), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-2)

class DS(torch.utils.data.Dataset):
    def __len__(self):
        return 128

    def __getitem__(self, i):
        return torch.randn(8), torch.randint(0, 2, (1,)).item()

trainer = pl.Trainer(max_epochs=2, logger=False, callbacks=[M().checkpoint_callback], enable_model_summary=False)
trainer.fit(M(), torch.utils.data.DataLoader(DS(), batch_size=32))
print('done')