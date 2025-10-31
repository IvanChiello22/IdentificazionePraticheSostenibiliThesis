# 30) PyTorch Lightning — T19 NEGATIVO: enable_checkpointing=False, nessun ModelCheckpoint
import torch, torch.nn as nn, pytorch_lightning as pl
import torch.nn.utils.prune as prune

class M(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,2))
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,x): return self.net(x)
    def training_step(self,batch,batch_idx):
        x,y = batch; return self.criterion(self(x), y)
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), 1e-2)

class DS(torch.utils.data.Dataset):
    def __len__(self): return 128
    def __getitem__(self,i): return torch.randn(8), torch.randint(0,2,(1,)).item()

def apply_energy_aware_pruning(model, amount=0.2):
    linear_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not linear_modules:
        return
    _, largest_module = max(linear_modules, key=lambda nm: nm[1].weight.numel())
    prune.l1_unstructured(largest_module, name='weight', amount=amount)

model = M()
apply_energy_aware_pruning(model, amount=0.2)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints", save_top_k=1)
trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=True, enable_model_summary=False, callbacks=[checkpoint_callback])
trainer.fit(model, torch.utils.data.DataLoader(DS(), batch_size=32))
print('done')