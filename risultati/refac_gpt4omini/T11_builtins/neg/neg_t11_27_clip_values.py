x = [-5, -1, 0, 3, 10]
low, high = -2, 4
y = [max(low, min(v, high)) for v in x]
print(y)