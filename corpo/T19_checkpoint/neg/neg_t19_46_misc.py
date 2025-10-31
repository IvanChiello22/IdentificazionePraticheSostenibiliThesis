# 46) Pure Python — T19 NEGATIVO: logistic regression SGD senza salvataggi
import random, math
w = [0.0]*5; b = 0.0
for e in range(3):
    for _ in range(64):
        x = [random.random() for _ in range(5)]; y = random.randint(0,1)
        z = sum(w[j]*x[j] for j in range(5)) + b
        p = 1/(1+math.exp(-z)); err = p - y
        for j in range(5): w[j] -= 0.05*err*x[j]
        b -= 0.05*err
print('done')
