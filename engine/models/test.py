import numpy as np


def mean_squared_error(output, expected):
    """
    Method to calculate loss using mean squared error \n
    - Expected output should be provided using one hot encoding and have the same shape as the output
    - Should only be used with a probability distribution (i.e. softmax)
    """
    return np.mean(np.square(output - expected))


np.set_printoptions(precision=4, suppress=True)

arr1 = np.array([0.2, 0.6, 0.2])
arr2 = np.array([0.7, 0.1, 0.2])
arr3 = np.array([[0.2, 0.6, 0.2], [0.7, 0.1, 0.2]])
expected = np.array([[0, 1, 0], [0, 1, 0]])
print(mean_squared_error(arr1, expected[0]))
print(np.mean(np.square(arr3 - expected), axis=1))


# print(cross_entropy(arr1, expected))
# print(cross_entropy(arr2, expected))

# arr = np.array([[-1, 0.25, 1.2], [3, 5.6, 9]])
# arr = np.array([-1, 0.25, 1.2])
# s = activation_softmax(arr)
# print(s)
# print(np.sum(s))

# print(activation_tanh(arr))
# print(activation_tanh_singular(0.6))

# print(activation_sigmoid(arr))
#
# for a in arr:
#     for i in a:
#         print(activation_sigmoid_singular(i), end="")
#     print("", end="\n")
