import csv 
import numpy as np

class Dataloader:
    def __init__(self, train_size, test_size, batch_size):

        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size

        with open('./dataset/mnist_train.csv', newline='') as f:
            reader = csv.reader(f)

            X = []
            y = []
            test_X = []
            test_y = []
            
            i = 0
            j = 0
            for row in reader:
                data = list(row)

                if i == train_size:
                    test_y.append(int(data.pop(0)))
                    test_X.append(data)
                    if j == test_size:
                        break
                    j += 1
                    continue

                y.append(int(data.pop(0)))
                X.append(data)
                i += 1

            self.X = np.array(X).astype(float) / 255
            self.y = np.array([np.eye(10)[i] for i in y])
            self.test_X = np.array(test_X).astype(float) / 255
            self.test_y = np.array([np.eye(10)[i] for i in y]) 
    
    def load_batch(self):
        random_choices = (self.train_size* np.random.rand(self.batch_size)).astype(int)
        return self.X[random_choices], self.y[random_choices]
    
    def load_test(self):
        random_choices = (self.test_size* np.random.rand(self.batch_size)).astype(int)
        return self.test_X[random_choices], self.test_y[random_choices]

if __name__ == "__main__":
    train_size = 2000
    test_size = 100
    batch_size = 128
    iteration = 20

    dataloader = Dataloader(train_size, test_size, batch_size)

    batch_1 = dataloader.load_batch()
    batch_2 = dataloader.load_batch()
    print(batch_1[1][0])
    print(batch_2[1][0])