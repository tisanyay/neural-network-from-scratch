import numpy as np
from dataloader import Dataloader
from metrics import mse, accuracy
from components import Softmax, ReLU, Linear

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

