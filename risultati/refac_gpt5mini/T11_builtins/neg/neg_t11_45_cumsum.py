from itertools import accumulate

x = [1, 2, 3, 4, 5]
y = list(accumulate(x))
s = y[-1] if y else 0
print(y)