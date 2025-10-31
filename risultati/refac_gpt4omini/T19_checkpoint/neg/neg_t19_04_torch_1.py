import torch, torch.nn as nn

lstm = nn.LSTM(12, 24, batch_first=True)
head = nn.Linear(24, 3)
opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), 1e-3)

# Aggiunta di checkpoint per salvare lo stato del modello
checkpoint_path = 'model_checkpoint.pth'

for e in range(3):
    x = torch.randn(10, 7, 12)
    y = torch.randint(0, 3, (10,))
    out, _ = lstm(x)
    logits = head(out[:, -1])
    loss = nn.CrossEntropyLoss()(logits, y)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # Salvataggio del checkpoint
    torch.save({
        'epoch': e,
        'model_state_dict': lstm.state_dict(),
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss.item(),
    }, checkpoint_path)

print('done')