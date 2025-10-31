# 45) Somma cumulativa (prefix sum)
x = [1, 2, 3, 4, 5]
y = []
s = 0
for v in x:
    s += v
    y.append(s)
print(y)
