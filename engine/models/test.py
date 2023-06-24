import numpy as np

#
# def activation_sigmoid(inputs):
#     """Sigmoid activation function"""
#     # TODO: Fix vanishing gradient problem
#     # TODO: Optimize
#     # return [1 / (1 + np.exp(-x)) for x in inputs]
#
#     # return np.exp(-inputs) + 1
#     return 1 / (np.exp(-inputs) + 1)
#
#
# np.set_printoptions(precision=4, suppress=True)
#
# arr1 = np.array([0.2, 0.6, 0.2])
# arr2 = np.array([0.7, 0.1, 0.2])
# arr3 = np.array([[0.2, 0.6, 0.2], [0.7, 0.1, 0.2]])
# expected = np.array([[0, 1, 0], [0, 1, 0]])
#
#
# print(activation_sigmoid(arr1))
# print(activation_sigmoid(arr3))


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

a = np.array([[1], [2], [3], [4]])
a2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

a3 = np.array([[2], [3], [4], [5]])
a4 = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])
print(np.greater(a4, 4) * 1)

