# 23) BiLSTM — T15 NEGATIVO: parametri completi
import torch, torch.nn as nn
import torch.nn.utils.prune as prune

lstm = nn.LSTM(16, 32, num_layers=2, bidirectional=True, batch_first=True)

# Apply energy-aware pruning strategy (no-op amount=0.0 to preserve behavior while enabling pruning hooks)
prune.global_unstructured(
    [(lstm, name) for name in [
        'weight_ih_l0', 'weight_hh_l0',
        'weight_ih_l1', 'weight_hh_l1',
        'weight_ih_l0_reverse', 'weight_hh_l0_reverse',
        'weight_ih_l1_reverse', 'weight_hh_l1_reverse'
    ]],
    pruning_method=prune.L1Unstructured,
    amount=0.0
)

x = torch.randn(5,4,16); y,_ = lstm(x)

# Save a checkpoint of the model state (useful during training to avoid knowledge loss)
torch.save({'model_state_dict': lstm.state_dict()}, 'checkpoint.pth')

print(y.shape)