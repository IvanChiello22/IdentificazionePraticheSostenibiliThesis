from collections import Counter

arr = [1, 2, 2, 3, 3, 3, 4]
counts = Counter(arr)
best = None
for k in counts:
    if best is None or counts[k] > counts[best]:
        best = k
print(best)