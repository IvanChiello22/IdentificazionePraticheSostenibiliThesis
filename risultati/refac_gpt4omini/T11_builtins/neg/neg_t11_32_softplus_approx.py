x = [-1.0, 0.0, 1.0]
y = [(v if v > 0 else 0.0) + 0.1 for v in x]
print(y)