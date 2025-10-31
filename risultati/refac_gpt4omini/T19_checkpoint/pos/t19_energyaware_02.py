import torch
import torch.nn as nn
import torch.optim as optim
import os

torch.manual_seed(0)

def create_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))

def create_optimizer(model):
    return optim.SGD(model.parameters(), 0.1)

def save_checkpoint(epoch, loss, model, filename):
    torch.save({'e': epoch, 'loss': loss, 'model': model.state_dict()}, filename)

def main():
    net = create_model()
    opt = create_optimizer(net)
    best = 1e9
    patience, wait = 2, 0
    os.makedirs('ckpt', exist_ok=True)

    for epoch in range(1, 13):
        x = torch.randn(64, 8)
        y = torch.randint(0, 2, (64,))
        loss = nn.CrossEntropyLoss()(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if loss.item() + 1e-6 < best:
            best = loss.item()
            wait = 0
            save_checkpoint(epoch, best, net, 'ckpt/best.pt')
        else:
            wait += 1
            if wait >= patience:
                save_checkpoint(epoch, loss.item(), net, f'ckpt/plateau_e{epoch:02d}.pt')
                wait = 0

    print('done')

if __name__ == "__main__":
    main()