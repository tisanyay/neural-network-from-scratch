import numpy as np

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
