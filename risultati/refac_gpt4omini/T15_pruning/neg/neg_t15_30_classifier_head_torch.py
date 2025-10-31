import torch
import torch.nn as nn

backbone = nn.Linear(64, 64)
head = nn.Sequential(nn.ReLU(), nn.Linear(64, 10))

# Utilizzo di checkpoint per salvare lo stato del modello durante l'addestramento
checkpoint = None

def forward_pass(input_tensor):
    global checkpoint
    output = head(backbone(input_tensor))
    checkpoint = output  # Salva il checkpoint
    return output

x = torch.randn(3, 64)
print(forward_pass(x).shape)