# Visualization of binary classification using logistic regression
# Includes change of cost and the data

import numpy as np
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt

def gen_rand_data(datasize, seed):
    '''Generates a random dataset used for binary classification on two dimensions.'''
    np.random.seed(seed) # fix a seed
    # Generate random points
    x = np.random.normal(4, 2, datasize)
    y = np.random.uniform(0, 10, datasize)
    # Generate a random separation
    m = np.random.uniform(0.5, 2)
    b = np.random.uniform(-2, 2)
    print(m, b)
    # Split datapoints
    sep = y > m * x + b
    return np.vstack((np.ones(datasize), x, y)).T, sep.astype(int)

def get_real_theta(theta):
    m = -theta[1] / theta[2]
    b = -theta[0] / theta[2]
    return m, b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(x, theta):
    return sigmoid(theta @ x.T)

def cost_gradient(theta, x, y, datasize):
    return (x.T @ (h(theta, x) - y)) / datasize # vectorized implementation

def fit(x, y, alpha=0.1, threashold=0.1, maxIter=1e6):
    theta = np.zeros(3, dtype='float128') # initialize theta
    # Gradient descent
    grad = cost_gradient(theta, x, y, datasize) # initialize gradient
    epoch = 0
    while abs(grad).max() > threashold:
        theta -= alpha * grad # perform gradient descent
        yield theta
        grad = cost_gradient(theta, x, y, datasize) # update gradient
        epoch += 1
        if epoch >= maxIter:
            break

datasize = 100
x, y = gen_rand_data(datasize, 2334)
sep = y.astype(bool) # separating the two classes

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.plot(*x[sep].T[1:, :], 'bx'); # positive class
axis.plot(*x[~sep].T[1:, :], 'g+'); # negative class
plt.xlim(-1, 9)
plt.ylim(-1, 11)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ion()

for theta in fit(x, y, alpha=0.03, threashold=0.1, maxIter=1e6):
    print(theta)
    m, b = get_real_theta(theta)
    xlin = np.linspace(-1, 9)
    boundary = axis.plot(xlin, m * xlin + b, 'r-')
    plt.pause(0.1)
    axis.lines.remove(boundary[0])

input('Done')
