# 26) Batch iterator semplice
data = list(range(10))
bs = 4
batches = []
for i in range(0, len(data), bs):
    batches.append(data[i:i+bs])
print(batches)
