# 48) Classificatore k-NN (k=3) naive su punti 2D
X = [(0,0),(1,0),(0,1),(1,1),(2,2)]
y = [0,0,0,1,1]
query = (0.9, 0.2)
k = 3
# distanza euclidea
dists = []
for i, (a,b) in enumerate(X):
    dx = a - query[0]
    dy = b - query[1]
    d = (dx*dx + dy*dy) ** 0.5
    dists.append((d, i))
# ordina per distanza (selection sort)
for i in range(len(dists)):
    best = i
    for j in range(i+1, len(dists)):
        if dists[j][0] < dists[best][0]:
            best = j
    dists[i], dists[best] = dists[best], dists[i]
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
