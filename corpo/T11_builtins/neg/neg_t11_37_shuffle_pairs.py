# 37) Shuffle di coppie (x,y) in parallelo
x = [1,2,3,4,5]
y = [10,20,30,40,50]
for i in range(len(x)-1, 0, -1):
    j = (i*29 + 7) % (i+1)
    x[i], x[j] = x[j], x[i]
    y[i], y[j] = y[j], y[i]
print(list(zip(x,y)))
