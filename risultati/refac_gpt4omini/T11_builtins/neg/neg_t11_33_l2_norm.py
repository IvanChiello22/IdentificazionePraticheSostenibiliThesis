import math

# 33) Norma L2 di un vettore
x = [3.0, 4.0]
norm = math.sqrt(sum(v * v for v in x))
print(norm)