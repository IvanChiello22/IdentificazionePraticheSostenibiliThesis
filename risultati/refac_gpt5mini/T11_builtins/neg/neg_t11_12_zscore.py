import statistics

# 12) Z-score (senza librerie)
x = [1.0, 2.0, 3.0, 4.0]
m = statistics.mean(x)
var = statistics.pvariance(x)
std = statistics.pstdev(x)
z = [(v - m)/(std or 1.0) for v in x]
print(z)