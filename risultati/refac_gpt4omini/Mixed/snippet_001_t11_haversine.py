import numpy as np

R = 6371.0
lat1 = np.deg2rad(np.random.uniform(-90, 90, (1000, 1)))
lon1 = np.deg2rad(np.random.uniform(-180, 180, (1000, 1)))
lat2 = np.deg2rad(np.random.uniform(-90, 90, (1, 1200)))
lon2 = np.deg2rad(np.random.uniform(-180, 180, (1, 1200)))

dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
D = R * c

print(D.shape, D.mean())