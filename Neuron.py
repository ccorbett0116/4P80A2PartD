import numpy as np


class Neuron:
    def __init__(self):
        self.bias = 0

    # sigmoid activation function
    def activate(self, inputs, weights):
        z = np.dot(inputs, weights) + self.bias
        return 1 / (1 + np.exp(-z))


def derivative(x):
    return x * (1 - x)

class MomentumNeuron(Neuron):
    def __init__(self, momentum_rate=0.9):
        super().__init__()
        self.momentum_rate = momentum_rate
        self.prev_delta_bias = 0

    def update_bias(self, delta_bias):
        total_delta_bias = delta_bias + self.momentum_rate * self.prev_delta_bias
        self.bias -= total_delta_bias
        self.prev_delta_bias = total_delta_bias

    def activate(self, inputs, weights):
        z = np.dot(inputs, weights) + self.bias
        return 1 / (1 + np.exp(-z))