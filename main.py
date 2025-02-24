import numpy as np
from dataloader import Dataloader
from metrics import mse, accuracy
from components import Softmax, ReLU, Linear

train_size = 10000
batch_size = 248
test_size = 100
iterations = 2000
alpha = 1

class NeuralNetwork:
    def __init__(self):
        self.components = [
            Linear(28*28, 10),
            ReLU(),
            Linear(10, 10),
            ReLU(),
            Linear(10, 10),
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

nn = NeuralNetwork()
dataloader = Dataloader(train_size, test_size, batch_size)

for i in range(iterations):
    X, y = dataloader.load_batch()
    prediction = nn.feed_forward(X)
    nn.backpropagate(y, alpha)
    
    if i % 10 == 0:
        print(str(i) + 'th iteration: ' + str(round(accuracy(prediction, y), 4)) + "%, " + str(round(mse(prediction, y), 4)))

test_X, test_y = dataloader.load_test()
prediction = nn.feed_forward(test_X)
print()
print('test: ')
print(str(i) + 'th iteration: ' + str(round(accuracy(prediction, test_y), 4)) + "%, " + str(round(mse(prediction, test_y), 4)))