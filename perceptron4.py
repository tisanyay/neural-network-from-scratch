import numpy as np
from dataloader import Dataloader

def mse(X, y):
    return np.sum(np.square(X-y)) / (len(X[0]) * 10)

def accuracy(X, y):
    predictions = np.argmax(X, axis=1)
    y = np.argmax(y, axis=1)

    return ((np.count_nonzero(predictions == y)) / len(predictions)) * 100

class Softmax:
    def feed_forward(self, x):
        x = x.T
        s = np.exp(x) / sum(np.exp(x))
        return s.T
    
    def derivative(self, X):
        return X
    
    def update_gradients(self, alpha):
        return

class ReLU:
    def __init__(self):
        self.output = 0

    def feed_forward(self, X):
        self.output = np.maximum(0, X)
        return self.output
    
    def derivative(self, error):
        return error * ((self.output > 0) * 1) 

    def update_gradients(self, alpha):
        return 

class Linear:
    def __init__(self, input, output):
        self.weights = np.random.randn(input, output) 
        self.biases = np.random.randn(1, output) 
        self.input = None
        self.m_gradient = None
        self.b_gradient = None 
    
    def feed_forward(self, X):
        self.input = X
        return X @ self.weights + self.biases
    
    def derivative(self, error):
        self.b_gradient = 1/1000 * np.sum(error, axis=0)
        self.m_gradient = 1/1000 * self.input.T @ error
        return error @ self.weights.T
    
    def update_gradients(self, alpha):
        self.weights -= alpha * self.m_gradient
        self.biases -= alpha * self.b_gradient
        return

class NeuralNetwork:
    def __init__(self):
        self.components = [
            Linear(28*28, 128),
            ReLU(),
            Linear(128, 16),
            ReLU(),
            # Linear(16, 16),
            # ReLU(),
            # Linear(16, 16),
            # ReLU(),
            Linear(16, 10),
            Softmax(),
        ]
        self.outputs = []

    def feed_forward(self, X):
        self.outputs.append(X)
        for component in self.components:
            X = component.feed_forward(X)
            self.outputs.append(X)
        return X
    
    def backpropagate(self, y, alpha):
        error = self.outputs.pop() - y
        for component in self.components[::-1]:
            error = component.derivative(error)
            component.update_gradients(alpha) 

train_size = 5000
batch_size = 248
test_size = 100
iterations = 200
alpha = 1e-2

nn = NeuralNetwork()
dataloader = Dataloader(train_size, test_size, batch_size)

for i in range(iterations):
    X, y = dataloader.load_batch()
    prediction = nn.feed_forward(X)
    nn.backpropagate(y, alpha)
    
    if i % 10 == 0:
        print(str(i) + 'th iteration: ' + str(round(accuracy(prediction, y), 4)) + "%")

