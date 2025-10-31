points = [(1,2),(1,4),(1,0),(10,2),(10,4),(10,0)]
centroids = [(1,2),(10,2)]
k = len(centroids)
assign = [0]*len(points)

# Assegnazione
for i, p in enumerate(points):
    best_c = 0
    best_d = float('inf')
    for c_idx, c in enumerate(centroids):
        dx = p[0]-c[0]
        dy = p[1]-c[1]
        d = dx*dx + dy*dy
        if d < best_d:
            best_d = d
            best_c = c_idx
    assign[i] = best_c

# Ricalcolo centroidi
new_centroids = [(0.0, 0.0) for _ in range(k)]
counts = [0]*k
for (px, py), a in zip(points, assign):
    cx, cy = new_centroids[a]
    new_centroids[a] = (cx + px, cy + py)
    counts[a] += 1
for i in range(k):
    if counts[i] > 0:
        cx, cy = new_centroids[i]
        new_centroids[i] = (cx / counts[i], cy / counts[i])
print(assign, new_centroids)