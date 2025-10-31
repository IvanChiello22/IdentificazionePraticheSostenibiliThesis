# 43) Quantile p (ordinamento + indice)
arr = [10, 2, 8, 4, 6, 0, 3]
p = 0.75  # quantile 75%
arr = sorted(arr)
idx = int(p * (len(arr)-1))
print(arr[idx])