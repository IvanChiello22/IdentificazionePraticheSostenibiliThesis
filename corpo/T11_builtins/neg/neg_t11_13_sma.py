# 13) Media mobile semplice
x = [1,2,3,4,5,6]
w = 3
res = []
for i in range(len(x)-w+1):
    s = 0
    for j in range(w):
        s += x[i+j]
    res.append(s/float(w))
print(res)
