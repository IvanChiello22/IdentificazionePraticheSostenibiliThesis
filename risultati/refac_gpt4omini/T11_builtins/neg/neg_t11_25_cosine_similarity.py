from math import sqrt

a = [1.0, 2.0, 3.0]
b = [2.0, 0.0, 1.0]

dot = sum(x * y for x, y in zip(a, b))
na = sum(x * x for x in a)
nb = sum(y * y for y in b)

sim = dot / (sqrt(na) * sqrt(nb) if na and nb else 1.0)
print(sim)