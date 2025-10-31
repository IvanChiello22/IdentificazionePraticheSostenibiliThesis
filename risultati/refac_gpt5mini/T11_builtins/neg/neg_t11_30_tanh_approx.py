# 30) Tanh approssimata
x = [-2.0, -1.0, 0.0, 1.0, 2.0]
y = [v / (1 + abs(v)) for v in x]  # approssimazione rozza
print(y)