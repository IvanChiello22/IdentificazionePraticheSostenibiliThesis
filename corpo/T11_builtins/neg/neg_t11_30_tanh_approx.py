# 30) Tanh approssimata
x = [-2.0, -1.0, 0.0, 1.0, 2.0]
y = []
for v in x:
    y.append(v/(1+abs(v)))  # approssimazione rozza
print(y)
