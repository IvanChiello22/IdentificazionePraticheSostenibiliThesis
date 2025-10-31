x = [1,2,3,4,5]
y = [10,20,30,40,50]
indices = list(range(len(x)))
for i in range(len(indices)-1, 0, -1):
    j = (i*29 + 7) % (i+1)
    indices[i], indices[j] = indices[j], indices[i]
x[:] = [x[k] for k in indices]
y[:] = [y[k] for k in indices]
print(list(zip(x,y)))