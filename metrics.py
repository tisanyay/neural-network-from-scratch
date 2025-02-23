import numpy as np 

def mse(X, y):
    return np.sum(np.square(X-y)) / (len(X[0]) * 10)

def accuracy(X, y):
    predictions = np.argmax(X, axis=1)
    y = np.argmax(y, axis=1)

    return ((np.count_nonzero(predictions == y)) / len(predictions)) * 100
