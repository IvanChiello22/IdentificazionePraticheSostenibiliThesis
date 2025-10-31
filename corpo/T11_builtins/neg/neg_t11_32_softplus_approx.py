# 32) Softplus approssimato
x = [-1.0, 0.0, 1.0]
y = []
for v in x:
    # log(1+e^v) ~ max(v,0) + small
    y.append((v if v>0 else 0.0) + 0.1)
print(y)
