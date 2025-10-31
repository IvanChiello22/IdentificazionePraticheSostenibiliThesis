labels = [0, 0, 1, 2, 2, 2, 1, 1, 1, 1]
counts = {y: labels.count(y) for y in set(labels)}
weights = {c: len(labels) / (cnt * len(counts)) for c, cnt in counts.items()}
print(weights)