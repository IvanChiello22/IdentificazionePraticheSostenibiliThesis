# 17) Argmax
x = [0.1, 0.7, 0.2, 0.6]
best_i = 0
for i in range(1, len(x)):
    if x[i] > x[best_i]:
        best_i = i
print(best_i)
