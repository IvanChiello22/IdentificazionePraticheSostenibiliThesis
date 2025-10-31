import torch, torch.nn as nn

gru = nn.GRU(16, 32, num_layers=2, batch_first=True)
head = nn.Linear(32, 5)
opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), 1e-3)

# Aggiunta di checkpoint per salvare il modello durante l'allenamento
checkpoint_path = 'model_checkpoint.pth'

for e in range(2):
    x = torch.randn(12, 9, 16)
    y = torch.randint(0, 5, (12,))
    o, _ = gru(x)
    logits = head(o[:, -1])
    loss = nn.CrossEntropyLoss()(logits, y)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # Salvataggio del checkpoint
    torch.save({'epoch': e, 'model_state_dict': gru.state_dict(), 'head_state_dict': head.state_dict()}, checkpoint_path)

print('done')