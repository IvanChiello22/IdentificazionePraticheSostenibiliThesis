import torch
import torch.nn as nn

# Funzione per potare la rete neurale
def prune_network(model):
    # Potatura della rete a partire dallo strato più energivoro
    for layer in reversed(list(model.children())):
        if isinstance(layer, nn.Linear):
            # Esempio di potatura: rimuovere il 20% dei neuroni
            weight = layer.weight.data
            threshold = torch.quantile(weight.abs(), 0.8)
            layer.weight.data[weight.abs() < threshold] = 0

net = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))
x = torch.randn(4, 50)

# Eseguire la potatura prima della previsione
prune_network(net)
print(net(x).shape)