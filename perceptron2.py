import numpy as np
from dataloader import Dataloader

def softmax(x):
    x = x.T
    s = np.exp(x) / sum(np.exp(x))
    return s.T

def mse(X, y):
    return np.sum(np.square(X-y)) / (len(X[0]) * 10)

def accuracy(X, y):
    predictions = np.argmax(X, axis=1)
    y = np.argmax(y, axis=1)

    return ((np.count_nonzero(predictions == y)) / len(predictions)) * 100

def relu(X):
    return np.maximum(0, X)

def relu_derivative(X):
    return (X > 0) * 1

def feed_forward(X, weight1, weight2, bias1, bias2):
    output1 = X @ weight1
    output1 = output1 + bias1.T
    output1 = relu(output1)

    prediction = output1 @ weight2
    prediction = prediction + bias2.T
    prediction = prediction

    return prediction, output1

def backpropagate(prediction, X, y, output1, weight1, weight2, bias1, bias2, alpha):
    m = y.size
    error_prediction = prediction - y
    difference_weight2 = alpha * (output1.T @ error_prediction)
    difference_bias2 = alpha * ((1/m) * np.sum(error_prediction, axis=1))
    updated_weight2 = weight2 - difference_weight2
    updated_bias2 = bias2 - difference_bias2

    error_output1 = (weight2 @ error_prediction.T).T * relu_derivative(output1)
    difference_weight1 = alpha * (X.T @ error_output1) 
    difference_bias1 = alpha * ((1/m) * np.sum(error_output1, axis=1))
    updated_weight1 = weight1 - difference_weight1
    updated_bias1 = bias1 - difference_bias1

    return updated_weight1, updated_weight2, updated_bias1, updated_bias2

def gradient_descent():        
    train_size = 2000
    test_size = 100
    batch_size = 1
    iteration = 1000
    alpha = 0.0000001

    dataloader = Dataloader(train_size, test_size, batch_size)

    W1 = np.random.randn(784, 16) 
    b1 = np.random.randn(16, 1) 
    W2 = np.random.randn(16, 10)
    b2 = np.random.randn(10, 1) 

    for _ in range(iteration):
        X, y = dataloader.load_batch()
        prediction, output1 = feed_forward(X, W1, W2, b1, b2)
        
        # print(prediction)
        print(str(round(accuracy(prediction, y), 4)) + "%")

        W1, W2, b1, b2 = backpropagate(prediction, X, y, output1, W1, W2, b1, b2, alpha)

gradient_descent()
