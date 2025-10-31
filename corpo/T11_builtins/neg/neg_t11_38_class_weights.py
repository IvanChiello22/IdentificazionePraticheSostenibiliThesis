# 38) Bilanciamento pesi di classe (conteggio manuale)
labels = [0,0,1,2,2,2,1,1,1,1]
counts = {}
for y in labels:
    counts[y] = counts.get(y, 0) + 1
weights = {}
total = len(labels)
for c, cnt in counts.items():
    weights[c] = total / (cnt * len(counts))
print(weights)
