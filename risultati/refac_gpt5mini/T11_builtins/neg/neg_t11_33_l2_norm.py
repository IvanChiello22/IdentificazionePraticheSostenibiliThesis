import math
x = [3.0, 4.0]
s = sum(v * v for v in x)
norm = math.sqrt(s)
print(norm)