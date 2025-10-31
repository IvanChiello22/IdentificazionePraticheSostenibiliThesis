import os
import pickle

# 21) SGD step su linea (y = wx + b)
x = [0.0, 1.0, 2.0]
y = [1.0, 3.0, 5.0]
w, b = 0.0, 0.0
lr = 0.01

_checkpoint_file = 'sgd_checkpoint.pkl'
_start_epoch = 0
if os.path.exists(_checkpoint_file):
    try:
        with open(_checkpoint_file, 'rb') as _f:
            _ck = pickle.load(_f)
            if isinstance(_ck, dict) and {'epoch', 'w', 'b'}.issubset(_ck.keys()):
                w = _ck['w']
                b = _ck['b']
                _start_epoch = _ck['epoch'] + 1
    except Exception:
        pass

for epoch in range(_start_epoch, 10):
    dw = 0.0; db = 0.0
    for xi, yi in zip(x, y):
        pred = w*xi + b
        err = pred - yi
        dw += err*xi
        db += err
    w -= lr * dw / len(x)
    b -= lr * db / len(x)
    try:
        with open(_checkpoint_file, 'wb') as _f:
            pickle.dump({'epoch': epoch, 'w': w, 'b': b}, _f)
    except Exception:
        pass

print(w, b)