x = np.random.randn(100000); y = 0.5*x + np.random.randn(100000)*0.5
H, xedges, yedges = np.histogram2d(x, y, bins=64, range=[[-4,4],[-4,4]])
P = np.divide(H, H.sum(), out=np.zeros_like(H), where=H.sum()!=0)
print(P.shape, round(P.max(),4))