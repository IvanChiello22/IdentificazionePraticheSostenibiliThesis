import numpy as np
from collections import Counter

# 48) Classificatore k-NN (k=3) naive su punti 2D
X = np.array([(0,0),(1,0),(0,1),(1,1),(2,2)])
y = [0,0,0,1,1]
query = np.array((0.9, 0.2))
k = 3

# distanza euclidea
dists = np.linalg.norm(X - query, axis=1)
# ottieni gli indici dei k vicini più prossimi
nearest_indices = np.argsort(dists)[:k]
votes = Counter(y[idx] for idx in nearest_indices)
pred = votes.most_common(1)[0][0]

print(pred)