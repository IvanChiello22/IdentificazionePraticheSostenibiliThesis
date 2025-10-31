import random
import math
import os
import pickle

def save_model(weights, bias, filename='model_checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((weights, bias), f)

def load_model(filename='model_checkpoint.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None, None

def logistic_regression_sgd(epochs=3, batch_size=64, learning_rate=0.05):
    weights = [0.0] * 5
    bias = 0.0
    model_checkpoint = load_model()

    if model_checkpoint[0] is not None:
        weights, bias = model_checkpoint

    for e in range(epochs):
        for _ in range(batch_size):
            x = [random.random() for _ in range(5)]
            y = random.randint(0, 1)
            z = sum(weights[j] * x[j] for j in range(5)) + bias
            p = 1 / (1 + math.exp(-z))
            err = p - y
            for j in range(5):
                weights[j] -= learning_rate * err * x[j]
            bias -= learning_rate * err

        save_model(weights, bias)

    print('done')

logistic_regression_sgd()