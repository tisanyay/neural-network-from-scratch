import numpy as np
from dataloader import Dataloader

def mse(X, y):
    return np.sum(np.square(X-y)) / (len(X[0]) * 10)

def accuracy(X, y):
    predictions = np.argmax(X, axis=1)
    y = np.argmax(y, axis=1)

    return ((np.count_nonzero(predictions == y)) / len(predictions)) * 100

class Perceptron:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.l1 = np.random.rand(784, 16) * np.sqrt(2/784)
        self.b1 = np.zeros((batch_size, 16))
        self.l2 = np.random.rand(16, 16) * np.sqrt(2/16)
        self.b2 = np.zeros((batch_size, 16))
        self.l3 = np.random.rand(16, 10)  * np.sqrt(2/16)
        self.b3 = np.zeros((batch_size, 10))

        self.layers = [self.l1, self.l2, self.l3]
        self.biases = [self.b1, self.b2, self.b3]
        self.outputs = []
    
    def feed_forward(self, X):
        m, n = X.shape
        self.outputs = [X]

        for layer, bias in zip(self.layers, self.biases):
            X = X @ layer
            X = X + bias
            X = np.maximum(0, X)
            self.outputs.append(X)

        return X
    
    def backpropagate(self, y, alpha):
        prediction = self.outputs.pop()
        error =  (prediction - y) / y.shape[0]

        for output, layer, bias in zip(self.outputs[::-1], self.layers[::-1], self.biases[::-1]):
            layer -= alpha * (output.T @ error) 
            print('bias:', bias.shape)
            print('error:', error.shape)
            bias -= alpha * error 
            error = (np.maximum(0, layer @ error.T).T) 
        
train_size = 2000
test_size = 100
batch_size = 128
iteration = 20

dataloader = Dataloader(train_size, test_size, batch_size)
perceptron = Perceptron(batch_size)

for _ in range(iteration):
    X, y = dataloader.load_batch()
    predictions = perceptron.feed_forward(X)
    # print(mse(predictions, y))
    # print(str(round(accuracy(predictions, y), 4)) + "%")
    print(predictions)
    perceptron.backpropagate(y, 0.000001)


# test_X, test_y = dataloader.load_test()
# test_prediction = perceptron.feed_forward(test_X)
# print(str(accuracy(test_prediction, test_y)) + "%")