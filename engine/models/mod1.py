import numpy as np
from engine.util import HiddenLayer, Softmax, Loss

# model = [
#     HiddenLayer(2, 2),
#     Softmax(2, 1)
# ]
#
# inputs = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 1],
#     [1, 0],
# ], dtype=float)
#
# fpl0 = model[0].activation_relu(inputs)
# ol0 = model[1].activation_softmax(fpl0)
# print(ol0)

class Layer:
    def __init__(self, inputs, outputs):
        self.weights = 2 * np.random.rand(outputs, inputs) - 1
        self.biases = 2 * np.random.rand(outputs) - 1

    def calc(self, inputs):
        # TODO: fix biases
        return np.dot(inputs, self.weights.T) + self.biases

    def activation_relu(self, inputs):
        return np.maximum(inputs, 0)

    def activation_softmax(self, inputs):
        i_exp = np.exp(inputs)
        s = np.sum(i_exp, axis=1)
        return np.divide(i_exp.T, s).T

    def activation_sigmoid(self, inputs):
        return 1 / (np.exp(-inputs) + 1)

    @staticmethod
    def loss_mse(output, expected):
        squares = np.power(output - expected, 2)
        return np.mean(squares)


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected = np.array([[0], [1], [1], [0]])

l1 = Layer(2, 2)
ol1 = Layer(2, 1)

lr = 0.5
num_epochs = 500

def forward_prop():
    z1 = l1.calc(inputs)
    a1 = l1.activation_sigmoid(z1)
    z2 = ol1.calc(a1)
    a2 = ol1.activation_sigmoid(z2)
    return Layer.loss_mse(a2, expected), {"z1": z1, "a1": a1, "z2": z2, "a2": a2}


def back_prop(cache):

    dca2 = 2 * (cache["a2"] - expected)
    da2z2 = cache["a2"] * (1 - cache["a2"])

    # print("DCA2: \n", dca2)
    # print("DA2Z2: \n", da2z2)
    # print("CACHE A1: \n", cache["a1"])
    # print("Z2/A2 WEIGHTS: \n", ol1.weights)

    dw2 = np.expand_dims(np.mean(cache["a1"] * da2z2 * dca2, axis=0), axis=0)
    db1 = np.dot(np.mean(dca2 * da2z2), np.array([[1, 1]])).flatten()

    # print("DW2: \n", dw2)
    # print("WEIGHTS 2 \n", ol1.weights)
    # print("DB1: \n", db1)
    # print("BIASES: \n", l1.biases)

    dz2a1 = np.dot(dca2 * da2z2, ol1.weights)
    da1z1 = cache["a1"] * (1 - cache["a1"]) * dz2a1
    dw1 = np.dot(da1z1.T, inputs) / len(inputs)
    # print("DA1Z1: \n", da1z1)
    # print("INPUTS: \n", inputs)
    # print("WEIGHTS: \n", l1.weights)
    # print("DW1: \n", dw1)

    ol1.weights = dw2 * -1 * lr
    l1.biases = db1 * -1 * lr
    l1.weights = dw1 * -1 * lr


loss, c = forward_prop()
print("ORIGINAL LOSS: ", loss)

for i in range(0, num_epochs):
    back_prop(c)
    loss, c = forward_prop()

print("END LOSS: ", loss)


