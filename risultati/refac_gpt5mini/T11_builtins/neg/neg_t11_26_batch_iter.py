# 26) Batch iterator semplice
data = list(range(10))
bs = 4
batches = [data[i:i+bs] for i in range(0, len(data), bs)]
print(batches)