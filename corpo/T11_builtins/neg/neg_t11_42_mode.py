# 42) Moda di una lista (conteggio manuale)
arr = [1, 2, 2, 3, 3, 3, 4]
counts = {}
for v in arr:
    counts[v] = counts.get(v, 0) + 1
best = None
for k in counts:
    if best is None or counts[k] > counts[best]:
        best = k
print(best)
