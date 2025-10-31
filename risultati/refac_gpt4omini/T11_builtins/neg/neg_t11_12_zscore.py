x = [1.0, 2.0, 3.0, 4.0]
m = sum(x) / len(x)
var = sum((v - m) ** 2 for v in x) / len(x)
std = var ** 0.5
z = [(v - m) / std for v in x]
print(z)