logits = [2.0, 1.0, 0.1]
exps = [1 + z + (z*z)/2 for z in logits]
s = sum(exps)
probs = [e/s for e in exps]
print(probs)