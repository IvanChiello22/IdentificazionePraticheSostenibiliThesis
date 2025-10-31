import torch
import torch.nn as nn

vocab_size = 200
embedding_dim = 64
hidden_dim = 128

embedding_layer = nn.Embedding(vocab_size, embedding_dim)
lstm_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
output_layer = nn.Linear(hidden_dim, vocab_size)

input_data = torch.randint(0, vocab_size, (4, 20))
lstm_output, _ = lstm_layer(embedding_layer(input_data))
logits = output_layer(lstm_output)

print(logits.shape)