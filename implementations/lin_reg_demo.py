# Demonstration of Univariate Linear Regression
# Shows how the hypothesis changes with gradient descent
# Based on lin_reg_uni.py

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import time

def gen_data(theta=None, thetaRange=(-10, 10, -20, 20), dataSize=100, xMin=0, xMax=10, error=4):
    # Generate the correct answer for the data set
    theta = theta or np.array([np.random.uniform(*thetaRange[:2]), np.random.uniform(*thetaRange[2:])])
    # Generate the data set
    data, target = np.array([]), np.array([])
    for i in range(dataSize):
        x = np.random.uniform(xMin, xMax)
        y = theta[0] + theta[1] * x + np.random.uniform(-error, error)
        data = np.append(data, x)
        target = np.append(target, y)
    return theta, data, target

def J(theta, trainX, trainY, trainSize):
    return np.sum((theta.dot(trainX) - trainY) ** 2) / (2 * trainSize)

def partial_derivative(theta, trainX, trainY, trainSize):
    return (theta.dot(trainX) - trainY).dot(trainX.T) / trainSize

def fit(trainX, trainY, learningRate=0.1, threashold=1E-5):
    theta = np.array([0., 1.]) # Initialize theta
    trainSize = len(trainY)
    derivative = partial_derivative(theta, trainX, trainY, trainSize)
    # costs = np.array([])
    index = 0
    while abs(derivative).max() > threashold: # While gradient doesn't converge
        if index % 5 == 0:
            yield theta
        index += 1
        # costs = np.append(costs, J(theta, trainX, trainY, trainSize)) # Record costs
        theta -= learningRate * derivative # Perform gradient descent
        derivative = partial_derivative(theta, trainX, trainY, trainSize) # Update partial derivative

def standard_fit(trainX, trainY):
    trainX = trainX[1, :] # Remove the extra bias term
    model = np.polyfit(trainX, trainY, deg=1) # Degree of 1 (linear model)
    return model

def accuracy(theta, testX, testY):
    # Returns the R Squared value of the model
    explained = theta.dot(testX)
    mean = np.sum(testY) / len(testY)
    totSumSquares = np.sum((testY - mean) ** 2)
    resSumSquares = np.sum((testY - explained) ** 2)
    accuracy = 1 - resSumSquares / totSumSquares
    return accuracy

dataset = datasets.load_diabetes()
data, target = dataset.data, dataset.target
data = data[:, 2] # Take only one feature
dataSize = len(data)
data = np.vstack((np.ones(dataSize), data)) # Add an extra bias term
separation = int(dataSize * 0.8) # Percentage of training data set and testing data set
trainX, trainY = data[:, :separation], target[:separation]
testX, testY = data[:, separation:], target[separation:]

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.plot(trainX[1, :], trainY, 'bo', marker='+') # Plot test samples
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ion()

xMin = trainX[1, :].min()
xMax = trainX[1, :].max()
for hypothesis in fit(trainX, trainY, 1, 1):
    print(hypothesis)
    x = np.linspace(xMin, xMax)
    y = hypothesis.dot(np.vstack((np.ones(len(x)), x)))
    hypothesisLine = axis.plot(x, y, 'r-', lw=2) # Plot hypothesis line
    plt.pause(0.1)
    axis.lines.remove(hypothesisLine[0])

print('Fitted Model: y = {1}x {2} {0}'.format(abs(hypothesis[0]), hypothesis[1], ('-', '+')[hypothesis[0]>0]))
acc = accuracy(hypothesis, testX, testY)
print('Fitted Accuracy: {:.4f}'.format(acc))

input()
# Plot the cost of theta over iterations
# plt.plot(costs)
# plt.xlabel('number of iteration')
# plt.ylabel('$J(\\theta)$')
# plt.show()
# plt.clf()

# Plot the data, hypothesis and ground truth
# x = np.linspace(trainX[1, :].min(), trainX[1, :].max())
# plt.plot(trainX[1, :], trainY, 'bo', marker='+') # Plot test samples
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# y = hypothesis.dot(np.vstack((np.ones(len(x)), x)))
# plt.plot(x, y, 'r-', lw=2) # Plot hypothesis line
# plt.show()
