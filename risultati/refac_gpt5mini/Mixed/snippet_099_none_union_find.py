class DSU:
    def __init__(self,n):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        r = self.r
        p = self.p
        if r[ra] < r[rb]:
            ra, rb = rb, ra
        p[rb] = ra
        if r[ra] == r[rb]:
            r[ra] += 1
        return True
ds = DSU(1000)
for i in range(1000):
    ds.union(i, (i*3+1) % 1000)
print(sum(ds.find(i) == ds.find(0) for i in range(1000)))