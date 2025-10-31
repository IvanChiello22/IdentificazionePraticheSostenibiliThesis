# 46) Prodotto cumulativo
x = [1, 2, 3, 4]
y = []
p = 1
for v in x:
    p *= v
    y.append(p)
print(y)
