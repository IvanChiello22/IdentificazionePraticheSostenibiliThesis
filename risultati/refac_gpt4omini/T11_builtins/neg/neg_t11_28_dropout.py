x = [0.2, 0.5, 0.7, 0.9, 0.1]
p = 0.4
y = [v if ((i * 17 + 3) % 100) >= int(p * 100) else 0.0 for i, v in enumerate(x)]
print(y)