import torch
import torch.nn as nn
import torch.optim as optim

x = torch.randn(12, 4, 10)
lstm = nn.LSTM(10, 20, batch_first=False)
head = nn.Linear(20, 2)

# Checkpointing
checkpoint = {'lstm_state_dict': lstm.state_dict(), 'head_state_dict': head.state_dict()}

out, _ = lstm(x)
logits = head(out[-1])
y = torch.randint(0, 2, (4,))
opt = optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=1e-3)

loss = nn.CrossEntropyLoss()(logits, y)
loss.backward()
opt.step()

# Save checkpoint
torch.save(checkpoint, 'model_checkpoint.pth')

print("done")