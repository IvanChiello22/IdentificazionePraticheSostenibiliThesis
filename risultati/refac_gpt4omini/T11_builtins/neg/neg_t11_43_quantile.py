arr = [10, 2, 8, 4, 6, 0, 3]
p = 0.75  # quantile 75%
arr.sort()  # Utilizzo della funzione built-in per ordinare
idx = int(p * (len(arr)-1))
print(arr[idx])