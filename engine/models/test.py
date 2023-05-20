import numpy as np


def activation_sigmoid(inputs):
    return np.array([1 / (1 + np.exp(-i)) for i in inputs])


def activation_sigmoid_singular(i):
    return 1 / (1 + np.exp(-i))


def activation_tanh(inputs):
    ei, eni = np.exp(inputs), np.exp(-inputs)
    return (ei - eni) / (ei + eni)


def activation_tanh_singular(inputs):
    ei, eni = np.exp(inputs), np.exp(-inputs)
    return (ei - eni) / (ei + eni)


# def activation_softmax(inputs):
#     return np.array([np.exp(inputs)[i] / np.sum(np.exp(inputs), axis=1)[i] for i in range(len(inputs))])

def activation_softmax(inputs):
    return np.divide(np.exp(inputs).T, np.sum(np.exp(inputs).T, axis=0)).T


np.set_printoptions(precision=4, suppress=True)
# arr = np.array([[-1, 0.25, 1.2], [3, 5.6, 9]])
arr = np.array([-1, 0.25, 1.2])
s = activation_softmax(arr)
print(s)
print(np.sum(s))

# print(activation_tanh(arr))
# print(activation_tanh_singular(0.6))

# print(activation_sigmoid(arr))
#
# for a in arr:
#     for i in a:
#         print(activation_sigmoid_singular(i), end="")
#     print("", end="\n")
