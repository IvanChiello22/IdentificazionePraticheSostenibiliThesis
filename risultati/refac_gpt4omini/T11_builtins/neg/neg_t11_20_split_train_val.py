data = list(range(20))
train_size = int(len(data) * 0.7)
train, val = data[:train_size], data[train_size:]
print(len(train), len(val))