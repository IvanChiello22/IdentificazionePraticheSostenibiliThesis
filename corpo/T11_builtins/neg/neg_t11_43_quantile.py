# 43) Quantile p (ordinamento + indice)
arr = [10, 2, 8, 4, 6, 0, 3]
p = 0.75  # quantile 75%
# insertion sort
for i in range(1, len(arr)):
    key = arr[i]
    j = i - 1
    while j >= 0 and arr[j] > key:
        arr[j+1] = arr[j]
        j -= 1
    arr[j+1] = key
idx = int(p * (len(arr)-1))
print(arr[idx])
