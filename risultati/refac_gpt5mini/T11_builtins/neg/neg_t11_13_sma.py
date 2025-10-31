x = [1,2,3,4,5,6]
w = 3
res = []
p = [0]
for val in x:
    p.append(p[-1] + val)
for i in range(len(x)-w+1):
    res.append((p[i+w] - p[i]) / float(w))
print(res)