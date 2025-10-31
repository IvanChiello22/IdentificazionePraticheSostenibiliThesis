data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
m = sum(data) / len(data)
var = sum((x - m) ** 2 for x in data) / len(data)
print(var)