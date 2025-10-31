# Snippet 66: mixed set
# NONE — Union‑Find with path compression, pure Python
class DSU:
    def __init__(self,n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True

if __name__ == '__main__':
    ds = DSU(1000)
    for i in range(1000):
        ds.union(i, (i*3 + 1) % 1000)
    print(sum(ds.find(i) == ds.find(0) for i in range(1000)))