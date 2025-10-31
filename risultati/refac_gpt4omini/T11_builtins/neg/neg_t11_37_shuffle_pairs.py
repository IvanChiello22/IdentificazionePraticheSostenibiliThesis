x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 50]
import random
combined = list(zip(x, y))
random.shuffle(combined)
x[:], y[:] = zip(*combined)
print(list(zip(x, y)))