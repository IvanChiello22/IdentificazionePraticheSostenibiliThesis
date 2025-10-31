# 3) Varianza (senza librerie)
data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
s = 0.0
for x in data:
    s += x
m = s / len(data)
var = 0.0
for x in data:
    diff = x - m
    var += diff * diff
var /= len(data)
print(var)
