import torch, torch.nn as nn, torch.optim as optim

x = torch.randn(5, 7, 16)  # seq_len, batch, input_size
rnn = nn.RNN(16, 32, num_layers=1, nonlinearity='tanh')
head = nn.Linear(32, 4)

# Checkpointing
checkpoint_path = 'rnn_checkpoint.pth'
if os.path.exists(checkpoint_path):
    rnn.load_state_dict(torch.load(checkpoint_path))

out, h = rnn(x)
logits = head(out[-1])
y = torch.randint(0, 4, (7,))
opt = optim.SGD(list(rnn.parameters()) + list(head.parameters()), lr=0.05)

loss = nn.CrossEntropyLoss()(logits, y)
loss.backward()
opt.step()

# Save checkpoint
torch.save(rnn.state_dict(), checkpoint_path)

print("done")