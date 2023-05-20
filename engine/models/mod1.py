import numpy as np
import matplotlib as plt
from keras.datasets import mnist

class HiddenLayer:
    def __init__(self, inputs, outputs):
        self.weights = np.random.rand(inputs, outputs)
        self.biases = np.random.rand(outputs)

    def calc(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

    def activationReLU(self, inputs):
        return np.maximum(self.calc(inputs), 0)


class Softmax:
    def activationSoftmax(self, inputs):
        # Faulty code when working with batches of data
        return np.exp(inputs) / np.sum(np.exp(inputs))


i = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
l1 = HiddenLayer(6, 2)
l2 = HiddenLayer(2, 3)
o = Softmax()
ff1 = l1.activationReLU(i)
ff2 = l2.activationReLU(ff1)
output = o.activationSoftmax(ff2)

print(i, end="\n\n\n")
print(output)
print(np.sum(output))

"""
s1 = Softmax()
print(s1.activationSoftmax(np.array([3.2, 1.3, 0.2, 0.8])))

test = np.array([[1, 2, 3],[4, 5, 6]])
print(np.rot90(test))


        self.weights = np.array(
            [
                [0.88319202, 0.04940545],
                [0.85332121, 0.04842534],
                [0.5040203,  0.93099412],
                [0.27177196, 0.15110668],
                [0.21589597, 0.10006434],
                [0.97635271, 0.67171663]
            ]
        )
"""