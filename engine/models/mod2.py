import numpy as np
import numpy.random
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, inputs, outputs):
        self.weights = np.random.randn(outputs, inputs)
        self.biases = np.random.randn(outputs)

    def calc(self, inputs):
        # TODO: fix biases
        return np.dot(inputs, self.weights.T) + self.biases

    @staticmethod
    def activation_relu(inputs):
        return np.maximum(inputs, 0)

    @staticmethod
    def activation_softmax(inputs):
        i_exp = np.exp(inputs)
        s = np.sum(i_exp, axis=1)
        return np.divide(i_exp.T, s).T

    @staticmethod
    def activation_sigmoid(inputs):
        return 1 / (np.exp(-inputs) + 1)

    @staticmethod
    def loss_mse(output, expected):
        squares = (output - expected) ** 2
        return np.sum(squares) / len(expected)

class OutputLayer(Layer):
    def calc(self, inputs):
        # TODO: fix biases
        return np.dot(inputs, self.weights.T)


LEARNING_RATE = 0.01
NUM_EPOCHS = 20000

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Remember one hot? is this the correct input format?
expected = np.array([[0], [1], [1], [0]])

model = [
    Layer(2, 2),
    OutputLayer(2, 1)
]

def forward_prop():
    z0 = model[0].calc(inputs)
    a0 = Layer.activation_sigmoid(z0)
    z1 = model[1].calc(a0)
    a1 = Layer.activation_sigmoid(z1)
    loss = Layer.loss_mse(a1, expected)
    # print("FEEDFORWARD OUTPUT: ", a1)
    cache = {"z0": z0, "a0": a0, "z1": z1, "a1": a1}
    return loss, cache


def back_prop(cache):

    dca1 = 2 * (cache["a1"] - expected)
    da1z1 = np.multiply(cache["a1"], 1 - cache["a1"])
    dz1w1 = cache["a0"]

    dz1a0 = model[1].weights
    da0z0 = np.multiply(cache["z0"], 1 - cache["z0"])
    dz0w0 = inputs

    dcw1 = np.mean(dca1 * da1z1 * dz1w1, axis=0)
    model[1].weights -= (dcw1 * LEARNING_RATE)

    dcb1 = np.mean(dca1 * da1z1, axis=0)
    model[0].biases -= (dcb1 * LEARNING_RATE)

    dcw0 = np.mean(dca1 * da1z1 * dz1a0 * da0z0 * dz0w0, axis=0)
    model[0].weights -= (dcw0 * LEARNING_RATE)


l, c = forward_prop()
print("LOSS: ", l)
# print("CACHE: \n", cache)
losses = []
for i in range(0, NUM_EPOCHS):
    p = numpy.random.permutation(len(inputs))
    inputs = inputs[p]
    expected = expected[p]
    l, c = forward_prop()
    losses.append(l)
    back_prop(c)

print("LOSS: ", l)

plt.plot([x for x in range(0, len(losses))], [x for x in losses])
plt.title("what the hell")
plt.xlabel("num epochs")
plt.ylabel("mean network MSE loss")
plt.show()

