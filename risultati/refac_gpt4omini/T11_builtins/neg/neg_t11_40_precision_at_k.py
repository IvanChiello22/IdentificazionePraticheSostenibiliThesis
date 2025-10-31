y_true = [2, 0, 1]
y_scores = [[0.1, 0.3, 0.6],
            [0.7, 0.2, 0.1],
            [0.2, 0.8, 0.0]]
k = 2
hits = 0
for t, scores in zip(y_true, y_scores):
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if t in idxs[:k]:
        hits += 1
prec_at_k = hits / len(y_true)
print(prec_at_k)