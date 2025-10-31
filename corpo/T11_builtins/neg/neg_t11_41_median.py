# 41) Mediana di una lista (ordinamento bubble sort + selezione centrale)
arr = [7, 1, 3, 9, 5, 2]
# bubble sort
n = len(arr)
for i in range(n):
    for j in range(0, n-i-1):
        if arr[j] > arr[j+1]:
            arr[j], arr[j+1] = arr[j+1], arr[j]
mid = n // 2
if n % 2 == 1:
    median = arr[mid]
else:
    median = (arr[mid-1] + arr[mid]) / 2.0
print(median)
