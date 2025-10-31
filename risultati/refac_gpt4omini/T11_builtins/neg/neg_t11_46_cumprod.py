y = [1]
for v in x[1:]:
    y.append(y[-1] * v)
print(y)