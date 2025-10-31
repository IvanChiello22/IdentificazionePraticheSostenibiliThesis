import random, math, pickle, os
w = [0.0]*5; b = 0.0
checkpoint_file = 'checkpoint.pkl'
total_epochs = 3
start_epoch = 0
if os.path.exists(checkpoint_file):
    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'w' in data and 'b' in data and 'epoch' in data:
                w = data['w']; b = data['b']; start_epoch = data['epoch']
    except Exception:
        pass
for e in range(start_epoch, total_epochs):
    for _ in range(64):
        x = [random.random() for _ in range(5)]; y = random.randint(0,1)
        z = math.fsum(wj*xj for wj, xj in zip(w, x)) + b
        p = 1/(1+math.exp(-z)); err = p - y
        for j in range(5): w[j] -= 0.05*err*x[j]
        b -= 0.05*err
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'w': w, 'b': b, 'epoch': e+1}, f)
    except Exception:
        pass
print('done')