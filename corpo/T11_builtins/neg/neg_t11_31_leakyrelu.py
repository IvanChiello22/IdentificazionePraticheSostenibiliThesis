# 31) LeakyReLU
x = [-3, -1, 0, 2]
alpha = 0.1
y = []
for v in x:
    y.append(v if v > 0 else alpha*v)
print(y)
