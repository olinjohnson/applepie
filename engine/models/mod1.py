import numpy as np
import matplotlib as plt
from keras.datasets import mnist


class Layer:
    def __init__(self, inputs, outputs):
        self.weights = np.random.rand(inputs, outputs)
        self.biases = np.random.rand(outputs)

    def calc(self, inputs):
        """
        Method to calculate neuron values (inputs * weights + biases)
        before activation functions
        """
        return np.dot(inputs, self.weights) + self.biases


class HiddenLayer(Layer):
    """
    Child class of Layer to represent hidden layers of a neural network
    """
    def activation_relu(self, inputs):
        """ReLU activation function"""
        return np.maximum(self.calc(inputs), 0)

    def activation_sigmoid(self, inputs):
        """Sigmoid activation function"""
        # TODO: Fix vanishing gradient problem
        # TODO: Optimize
        return [1 / (1 + np.exp(-x)) for x in self.calc(inputs)]

    def activation_tanh(self, inputs):
        """tanh activation function"""
        # TODO: Fix vanishing gradient problem
        c = self.calc(inputs)
        ei, eni = np.exp(c), np.exp(-c)
        return (ei - eni) / (ei + eni)


class Softmax(Layer):
    """
    Child class of Layer to represent the output layer of a neural network
    """
    def activation_softmax(self, inputs):
        """Softmax activation function"""
        inext = np.exp(self.calc(inputs)).T
        return np.divide(inext, np.sum(inext, axis=0)).T


class Loss:
    """
    Class containing different loss functions
    # TODO: Implement other loss functions that might be beneficial
    """
    @staticmethod
    def cross_entropy(output, expected):
        """
        Method to calculate categorical cross entropy loss \n
        - Expected output should be provided using one hot encoding and have the same shape as the output
        """
        return -np.sum(np.array([expected[x] * np.log(output[x]) for x in range(len(expected))]))

    @staticmethod
    def mean_squared_error(output, expected):
        """
        Method to calculate loss using mean squared error \n
        - Expected output should be provided using one hot encoding and have the same shape as the output \n
        - Should only be used with a probability distribution (i.e. softmax)
        """
        return np.square(output - expected) / len(expected)


# i = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
i = np.array([1, 2, 3, 4, 5, 6])

model = [
    HiddenLayer(6, 2),
    HiddenLayer(2, 3),
    Softmax(3, 5),
]

ff1 = model[0].activation_relu(i)
ff2 = model[1].activation_relu(ff1)
o = model[2].activation_softmax(ff2)

np.set_printoptions(precision=4, suppress=True)
print(o)
print(np.sum(o))

# s1 = Softmax()
# print(s1.activationSoftmax(np.array([3.2, 1.3, 0.2, 0.8])))
#
# test = np.array([[1, 2, 3],[4, 5, 6]])
# print(np.rot90(test))
#
#
# self.weights = np.array(
# [
#         [0.88319202, 0.04940545],
#         [0.85332121, 0.04842534],
#         [0.5040203,  0.93099412],
#         [0.27177196, 0.15110668],
#         [0.21589597, 0.10006434],
#         [0.97635271, 0.67171663]
#     ]
# )
