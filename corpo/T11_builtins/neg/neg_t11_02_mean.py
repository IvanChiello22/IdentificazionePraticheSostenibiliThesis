# 2) Media di lista (loop)
data = [3, 5, 8, 10]
s = 0.0
n = 0
for x in data:
    s += x
    n += 1
mean = s / n if n else 0.0
print(mean)
