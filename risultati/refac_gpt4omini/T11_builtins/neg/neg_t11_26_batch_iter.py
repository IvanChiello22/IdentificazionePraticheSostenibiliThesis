batches = [data[i:i + bs] for i in range(0, len(data), bs)]
print(batches)