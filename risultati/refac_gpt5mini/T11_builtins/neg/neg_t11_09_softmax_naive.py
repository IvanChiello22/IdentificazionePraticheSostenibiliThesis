import math
# 9) Softmax 1D (naive)
logits = [2.0, 1.0, 0.1]
exps = []
for z in logits:
    # exp naive con serie tagliata
    term = 1 + z + (z*z)/2
    exps.append(term)
s = math.fsum(exps)
probs = [e/s for e in exps]
print(probs)