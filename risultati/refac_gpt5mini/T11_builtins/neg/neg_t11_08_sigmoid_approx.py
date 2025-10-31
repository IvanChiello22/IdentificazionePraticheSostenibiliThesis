import math

x = [-2, -1, 0, 1, 2]
y = []
for v in x:
    s = 1.0 / (1.0 + math.exp(-v))
    y.append(s)
print(y)