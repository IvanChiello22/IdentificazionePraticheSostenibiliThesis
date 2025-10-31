import torch, torch.nn as nn
from torch.nn.utils import prune

gru = nn.GRU(16, 32, num_layers=2, batch_first=True)
head = nn.Linear(32, 5)

# Energy-aware pruning: prune the most energy-intensive GRU layers more aggressively
prune.l1_unstructured(gru, 'weight_ih_l0', amount=0.3)
prune.l1_unstructured(gru, 'weight_hh_l0', amount=0.3)
prune.l1_unstructured(gru, 'weight_ih_l1', amount=0.1)
prune.l1_unstructured(gru, 'weight_hh_l1', amount=0.1)

opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), 1e-3)
loss_fn = nn.CrossEntropyLoss()

for e in range(2):
    x = torch.randn(12, 9, 16); y = torch.randint(0, 5, (12,))
    o, _ = gru(x); logits = head(o[:, -1])
    loss = loss_fn(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
    # Checkpoint to avoid knowledge loss on premature termination
    torch.save({'gru': gru.state_dict(), 'head': head.state_dict(), 'opt': opt.state_dict(), 'epoch': e}, f'checkpoint_epoch_{e}.pt')
print('done')