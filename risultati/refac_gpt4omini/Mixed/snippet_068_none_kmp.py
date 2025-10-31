def kmp_table(pattern):
    pattern_length = len(pattern)
    lps = [0] * pattern_length
    j = 0
    for i in range(1, pattern_length):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j
    return lps

def kmp_search(text, pattern):
    lps = kmp_table(pattern)
    j = 0
    results = []
    for i, char in enumerate(text):
        while j > 0 and char != pattern[j]:
            j = lps[j - 1]
        if char == pattern[j]:
            j += 1
        if j == len(pattern):
            results.append(i - j + 1)
            j = lps[j - 1]
    return results

print(kmp_search("ababcabcabababd", "ababd"))