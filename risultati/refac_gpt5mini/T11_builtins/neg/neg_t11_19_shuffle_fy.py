# 19) Shuffle (Fisher-Yates deterministico)
arr = [1,2,3,4,5,6]
for i in range(len(arr)-1, 0, -1):
    j = (i*37 + 13) % (i+1)  # pseudo-random deterministico
    arr[i], arr[j] = arr[j], arr[i]
print(arr)