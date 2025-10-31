import statistics

# 3) Varianza (senza librerie)
data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
var = statistics.pvariance(data)
print(var)