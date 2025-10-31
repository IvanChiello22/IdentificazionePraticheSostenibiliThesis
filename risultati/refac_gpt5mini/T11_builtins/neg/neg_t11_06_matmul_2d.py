# 6) Matmul 2D (triplo loop)
A = [[1, 2, 3],
     [4, 5, 6]]
B = [[7, 8],
     [9, 10],
     [11, 12]]
C = [[0]*len(B[0]) for _ in range(len(A))]
for i in range(len(A)):
    for k in range(len(B[0])):
        s = sum(A[i][j] * B[j][k] for j in range(len(B)))
        C[i][k] = s
print(C)