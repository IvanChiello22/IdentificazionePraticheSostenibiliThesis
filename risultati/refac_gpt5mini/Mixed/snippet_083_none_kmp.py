# Snippet 83: mixed set
# NONE — KMP string search (build prefix table), pure Python
def kmp_table(p: str):
    """Build longest proper prefix-suffix (lps) table for pattern p."""
    m = len(p)
    lps = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and p[i] != p[j]:
            j = lps[j - 1]
        if p[i] == p[j]:
            j += 1
            lps[i] = j
    return lps

def kmp_search(s: str, p: str):
    """Return list of start indices where pattern p is found in s using KMP."""
    lps = kmp_table(p)
    j = 0
    res = []
    for i, ch in enumerate(s):
        while j > 0 and ch != p[j]:
            j = lps[j - 1]
        if ch == p[j]:
            j += 1
        if j == len(p):
            res.append(i - j + 1)
            j = lps[j - 1]
    return res

print(kmp_search("ababcabcabababd", "ababd"))