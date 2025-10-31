x = [1, 2, 3, 4, 5, 6]
w = 3
res = [sum(x[i:i+w]) / float(w) for i in range(len(x) - w + 1)]
print(res)