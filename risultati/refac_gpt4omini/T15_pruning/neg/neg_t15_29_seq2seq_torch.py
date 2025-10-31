import torch, torch.nn as nn
import torch.optim as optim

enc = nn.LSTM(10, 20, batch_first=True)
dec = nn.LSTM(20, 20, batch_first=True)

# Checkpointing function
def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

x = torch.randn(2, 7, 10)
h, _ = enc(x)
y, _ = dec(h)

# Save checkpoint after processing
save_checkpoint(enc, 'encoder_checkpoint.pth')
save_checkpoint(dec, 'decoder_checkpoint.pth')

print(y.shape)