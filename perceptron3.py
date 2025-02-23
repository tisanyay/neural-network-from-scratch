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
    output1 = X @ weight1 + bias1
    output1 = relu(output1)
    prediction = output1 @ weight2
    prediction = softmax(prediction)
    return prediction, output1

def backpropagate(prediction, X, y, output1, weight2):
    m = y.size
    error_prediction = prediction - y
    difference_weight2 = 1/m * output1.T @ error_prediction  
    difference_bias2 = 1/m * np.sum(error_prediction, axis=0)

    error_output1 = error_prediction @ weight2.T  
    error_output1 = relu_derivative(error_output1) * error_output1
    # error_output1 = error_prediction @ weight2.T * relu_derivative(output1)
    # difference_weight1 = 1/m * X.T @ error_output1
    # difference_bias1 = 1/m * np.sum(error_output1, axis=0)
    difference_weight1 = 1/1000 * X.T @ error_output1
    difference_bias1 = 1/1000 * np.sum(error_output1, axis=0)

    return difference_weight1, difference_bias1, difference_weight2, difference_bias2

def update_params(weight1, difference_weight1, weight2, difference_weight2, bias1, difference_bias1, bias2, difference_bias2, alpha):
    weight1 -= alpha * difference_weight1
    weight2 -= alpha * difference_weight2
    bias1 -= alpha * difference_bias1
    bias2 -= alpha * difference_bias2
    return weight1, weight2, bias1, bias2

def get_predictions(predictions):
    return np.argmax(predictions, 0)

def get_accuracy(predictions, Y):
    Y = get_predictions(Y, 0)
    return np.sum(predictions == Y) / Y.size

def gradient_descent():
    train_size = 100
    test_size = 100
    batch_size = 100
    iterations = 1000
    alpha = 0.1

    dataloader = Dataloader(train_size, test_size, batch_size)

    W1 = np.random.randn(784, 12) 
    b1 = np.random.randn(1, 12) 
    W2 = np.random.randn(12, 10)
    b2 = np.random.randn(1, 10) 

    for i in range(iterations):
        X, y = dataloader.load_batch()
        prediction, output1 = feed_forward(X, W1, W2, b1, b2)
        dw1, db1, dw2, db2 = backpropagate(prediction, X, y, output1, W2)
        W1, W2, b1, b2 = update_params(W1, dw1, W2, dw2, b1, db1, b2, db2, alpha)

        print(str(round(accuracy(prediction, y), 4)) + "%")


gradient_descent()