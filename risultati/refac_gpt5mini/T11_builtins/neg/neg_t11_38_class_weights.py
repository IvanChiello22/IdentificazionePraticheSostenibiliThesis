# 38) Bilanciamento pesi di classe (conteggio manuale)
from collections import Counter

labels = [0,0,1,2,2,2,1,1,1,1]
counts = dict(Counter(labels))
weights = {}
total = len(labels)
for c, cnt in counts.items():
    weights[c] = total / (cnt * len(counts))
print(weights)