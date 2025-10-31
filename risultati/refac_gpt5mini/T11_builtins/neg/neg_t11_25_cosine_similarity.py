import math
a = [1.0, 2.0, 3.0]
b = [2.0, 0.0, 1.0]
dot = 0.0
na = 0.0
nb = 0.0
for i in range(len(a)):
    dot += a[i]*b[i]
    na += a[i]*a[i]
    nb += b[i]*b[i]
sim = dot / ((math.sqrt(na) or 1.0) * (math.sqrt(nb) or 1.0))
print(sim)