# 46) Prodotto cumulativo
from itertools import accumulate
import operator

x = [1, 2, 3, 4]
y = list(accumulate(x, operator.mul))
print(y)