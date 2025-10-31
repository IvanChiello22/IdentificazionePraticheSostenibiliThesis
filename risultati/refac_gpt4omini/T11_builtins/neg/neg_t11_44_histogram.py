data = [0.2, 1.1, 1.7, 2.0, 2.4, 2.9, 3.1]
bin_width = 1.0
min_v = 0.0
max_v = 3.0
num_bins = int((max_v - min_v) / bin_width)
bins = [0] * num_bins

for v in data:
    if min_v <= v < max_v:
        b = int((v - min_v) // bin_width)
        bins[b] += 1

print(bins)  # conteggi per bin [0-1), [1-2), [2-3)