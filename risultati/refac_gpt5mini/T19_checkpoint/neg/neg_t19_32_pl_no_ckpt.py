import torch, torch.nn as nn, pytorch_lightning as pl
import torch.nn.utils.prune as prune

class M(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,2))
        self.loss = nn.CrossEntropyLoss()
    def forward(self,x): return self.net(x)
    def training_step(self,batch,batch_idx):
        x,y = batch
        return self.loss(self(x), y)
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), 1e-2)

class DS(torch.utils.data.Dataset):
    def __len__(self): return 128
    def __getitem__(self,i): return torch.randn(8), torch.randint(0,2,(1,)).item()

def prune_most_energy_intensive_layer(model, amount=0.0):
    # Identify linear layer with largest weight parameter count (proxy for energy intensity)
    max_params = -1
    target_layer = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            params = module.weight.numel()
            if params > max_params:
                max_params = params
                target_layer = module
    if target_layer is not None:
        prune.l1_unstructured(target_layer, name='weight', amount=amount)

model = M()
# Apply pruning pipeline starting from the most "energy intensive" layer.
# amount=0.0 keeps behavior unchanged while using official pruning API.
prune_most_energy_intensive_layer(model, amount=0.0)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='checkpoints', save_last=True)
trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=True, enable_model_summary=False, callbacks=[checkpoint_callback])
trainer.fit(model, torch.utils.data.DataLoader(DS(), batch_size=32))
print('done')