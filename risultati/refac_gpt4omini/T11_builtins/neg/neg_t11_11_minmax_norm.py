x = [3, 7, 5, 9]
mn = min(x)
mx = max(x)
rng = (mx - mn) or 1.0
x_norm = [(v - mn) / rng for v in x]
print(x_norm)