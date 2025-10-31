# 40) Precision@k (k=2) manual
y_true = [2, 0, 1]
y_scores = [[0.1, 0.3, 0.6],
            [0.7, 0.2, 0.1],
            [0.2, 0.8, 0.0]]
k = 2
hits = 0
for t, scores in zip(y_true, y_scores):
    idxs = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    if t in idxs[:k]:
        hits += 1
prec_at_k = hits / len(y_true)
print(prec_at_k)