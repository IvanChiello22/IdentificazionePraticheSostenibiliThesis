x = [-3, -1, 0, 2]
alpha = 0.1
y = [v if v > 0 else alpha * v for v in x]
print(y)