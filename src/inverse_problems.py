# Q3.1

import numpy as np

from imagetools.optimisation import gradient_descent

# Define specific objective function, its gradient, and gradient descent step
obj = lambda x: 0.5 * (x[0] ** 2) + (x[1] ** 2)
grad = lambda x: np.array([x[0], 2.0 * x[1]])

# Perform gradient descent
x0 = np.array([1.0, 1.0])
lr = 0.0001
gradient_descent(obj, grad, x0, 0, 0.01, lr, 1000)
