# 30) Classifier head — T15 NEGATIVO: nessun pruning sul head
import torch, torch.nn as nn
from torch.nn.utils import prune
backbone = nn.Linear(64, 64)
head = nn.Sequential(nn.ReLU(), nn.Linear(64, 10))
# Energy-aware pruning: apply pruning starting from the most energy intensive layer (backbone).
# Use amount=0.0 to keep semantics identical while introducing the pruning step minimally.
prune.l1_unstructured(backbone, name='weight', amount=0.0)
# Checkpoint to avoid knowledge loss in case of premature termination.
torch.save({'backbone_state': backbone.state_dict(), 'head_state': head.state_dict()}, 'checkpoint.pth')
x = torch.randn(3,64); print(head(backbone(x)).shape)