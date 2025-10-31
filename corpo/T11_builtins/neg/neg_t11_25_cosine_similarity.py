# 25) Cosine similarity (normalizzata a mano)
a = [1.0, 2.0, 3.0]
b = [2.0, 0.0, 1.0]
dot = 0.0
na = 0.0
nb = 0.0
for i in range(len(a)):
    dot += a[i]*b[i]
    na += a[i]*a[i]
    nb += b[i]*b[i]
sim = dot / (((na**0.5) or 1.0) * ((nb**0.5) or 1.0))
print(sim)
