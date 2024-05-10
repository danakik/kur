import numpy as np
import matplotlib.pyplot as plt

def sigmoid(u):
    return 1 / (1 + np.exp(-0.5 * u))

def sigmoid_derivative(u):
    return sigmoid(u) * 0.5 * (1 - sigmoid(u))  

class NeuralNetwork:
    def __init__(self):
        self.W1_1 = np.array([0.1, -0.3, 0.4])
        self.W1_2 = np.array([-0.7, -0.1, 0.01])
        self.W2 = np.array([0.4, -0.2, 0.1])
        self.train_error = []
        self.test_error = []
        self.n = 0.1
        self.epochs = 100

    def train(self):
        for epoch in range(self.epochs):
            train_x1 = np.random.rand(1000)
            train_x2 = np.random.rand(1000)
            d = np.where((train_x2 < train_x1 + 0.5) & (train_x2 > train_x1 - 0.5), 0, 1)
            
            epoch_error = 0
            
            for i in range(len(train_x1)):
                U11 = self.W1_1[0] * 1 + self.W1_1[1] * train_x1[i] + self.W1_1[2] * train_x2[i]
                U12 = self.W1_2[0] * 1 + self.W1_2[1] * train_x1[i] + self.W1_2[2] * train_x2[i]
                Y1 = sigmoid(U11)
                Y2 = sigmoid(U12)
                U21 = self.W2[0] * 1 + self.W2[1] * Y1 + self.W2[2] * Y2
                Y = sigmoid(U21)
                
                E = d[i] - Y
                E_squared = E ** 2
                
                self.W1_1[0] += self.n * E * sigmoid_derivative(U21) * self.W2[1] * sigmoid_derivative(U11) 
                self.W1_1[1] += self.n * E * sigmoid_derivative(U21) * self.W2[1] * sigmoid_derivative(U11) * train_x1[i]
                self.W1_1[2] += self.n * E * sigmoid_derivative(U21) * self.W2[1] * sigmoid_derivative(U11) * train_x2[i]

                self.W1_2[0] += self.n * E * sigmoid_derivative(U21) * self.W2[2] * sigmoid_derivative(U12) 
                self.W1_2[1] += self.n * E * sigmoid_derivative(U21) * self.W2[2] * sigmoid_derivative(U12) * train_x1[i]
                self.W1_2[2] += self.n * E * sigmoid_derivative(U21) * self.W2[2] * sigmoid_derivative(U12) * train_x2[i]

                self.W2[0] += self.n * E * sigmoid_derivative(U21)
                self.W2[1] += self.n * E * sigmoid_derivative(U21) * Y1
                self.W2[2] += self.n * E * sigmoid_derivative(U21) * Y2

                epoch_error += E_squared
            
            self.train_error.append((epoch_error / 2) / len(train_x1))  

            test_x1 = np.array([0, 1, 0, 1])
            test_x2 = np.array([0, 0, 1, 1])
            test_d = np.array([0, 1, 1, 0])

            test_e = 0
            for i in range(len(test_x1)):
                U11 = self.W1_1[0] * 1 + self.W1_1[1] * test_x1[i] + self.W1_1[2] * test_x2[i]
                U12 = self.W1_2[0] * 1 + self.W1_2[1] * test_x1[i] + self.W1_2[2] * test_x2[i]
                Y1 = sigmoid(U11)
                Y2 = sigmoid(U12)
                U21 = self.W2[0] * 1 + self.W2[1] * Y1 + self.W2[2] * Y2
                Y = sigmoid(U21)
                E = test_d[i] - Y
                test_e += (E ** 2) 
            test_e = test_e / 2
            l = len(test_x1)
            self.test_error.append(test_e  / l)

    def plot_error(self):
        plt.plot(self.train_error, color ='black', label='Train')
        plt.plot(self.test_error, color ='blue', label='Test')
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title("Error")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.train()
    nn.plot_error()
