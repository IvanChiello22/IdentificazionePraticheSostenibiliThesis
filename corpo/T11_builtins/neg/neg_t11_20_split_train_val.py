# 20) Split train/val
data = list(range(20))
ratio = 0.7
cut = int(len(data)*ratio)
train = data[:cut]
val = data[cut:]
print(len(train), len(val))
