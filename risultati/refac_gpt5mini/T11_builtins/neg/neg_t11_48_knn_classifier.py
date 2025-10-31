# 48) Classificatore k-NN (k=3) naive su punti 2D
import math
X = [(0,0),(1,0),(0,1),(1,1),(2,2)]
y = [0,0,0,1,1]
query = (0.9, 0.2)
k = 3
# distanza euclidea
dists = []
for i, (a,b) in enumerate(X):
    dx = a - query[0]
    dy = b - query[1]
    d = math.hypot(dx, dy)
    dists.append((d, i))
# ordina per distanza usando built-in
dists = sorted(dists)
neighbors = dists[:k]
votes = {}
for _, idx in neighbors:
    c = y[idx]
    votes[c] = votes.get(c, 0) + 1
pred = None
for c in votes:
    if pred is None or votes[c] > votes[pred]:
        pred = c
print(pred)