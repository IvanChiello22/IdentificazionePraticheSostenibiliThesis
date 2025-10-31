arr = [7, 1, 3, 9, 5, 2]
arr.sort()
mid = len(arr) // 2
if len(arr) % 2 == 1:
    median = arr[mid]
else:
    median = (arr[mid-1] + arr[mid]) / 2.0
print(median)