# Snippet 87: mixed set
# NONE — KMP string search (build prefix table), pure Python
def kmp_table(p):
    m=len(p); lps=[0]*m; j=0
    for i in range(1,m):
        while j>0 and p[i]!=p[j]: j=lps[j-1]
        if p[i]==p[j]: j+=1; lps[i]=j
    return lps
def kmp_search(s,p):
    lps=kmp_table(p); res=[]
    if len(p)==0:
        raise IndexError('string index out of range')
    start=0
    while True:
        idx = s.find(p, start)
        if idx == -1:
            break
        res.append(idx)
        start = idx + 1
    return res
print(kmp_search("ababcabcabababd","ababd"))