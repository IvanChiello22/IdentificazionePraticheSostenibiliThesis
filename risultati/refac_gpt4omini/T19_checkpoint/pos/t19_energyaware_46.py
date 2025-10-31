import torch
import torch.nn as nn
import torch.optim as optim
import os

def load_checkpoint(model, optimizer, filepath):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        return checkpoint['e'] + 1
    return 1

def save_checkpoint(epoch, model, optimizer, filepath):
    torch.save({'e': epoch, 'model': model.state_dict(), 'opt': optimizer.state_dict()}, filepath)

def train_model(num_epochs, model, optimizer, checkpoint_filepath):
    start_epoch = load_checkpoint(model, optimizer, checkpoint_filepath)
    for epoch in range(start_epoch, start_epoch + num_epochs):
        x = torch.randn(16, 3)
        y = torch.randint(0, 2, (16,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_checkpoint(epoch, model, optimizer, checkpoint_filepath)
    print('done')

net = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
opt = optim.SGD(net.parameters(), 0.01)
train_model(5, net, opt, 'ckpt_last.pt')