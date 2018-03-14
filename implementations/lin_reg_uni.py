# Univariate Linear Regression
# Implementation using numpy, more efficient

import numpy as np
from sklearn import datasets

def partial_derivative(theta, trainX, trainY, trainSize):
    return (theta.dot(trainX) - trainY).dot(trainX.T) / trainSize

def fit(trainX, trainY, learningRate=0.1, threashold=1E-5):
    theta = np.array([0, 1], dtype=float) # Initialize theta
    trainSize = len(trainY)
    derivative = partial_derivative(theta, trainX, trainY, trainSize)
    while abs(derivative).max() > threashold: # While gradient doesn't converge
        theta -= learningRate * derivative # Perform gradient descent
        derivative = partial_derivative(theta, trainX, trainY, trainSize) # Update partial derivative
    return theta

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
dataSize = len(data)
data = data[:, 2] # Take only one feature
data = np.vstack((np.ones(dataSize), data)) # Add an extra bias term
separation = int(dataSize * 0.8) # Percentage of training data set and testing data set
trainX, trainY = data[:, :separation], target[:separation]
testX, testY = data[:, separation:], target[separation:]

hypothesis = standard_fit(trainX, trainY)
print('Fitted Model: y = {0}x {2} {1}'.format(hypothesis[0], abs(hypothesis[1]), ('-', '+')[hypothesis[1]>0]))
acc = accuracy(hypothesis[::-1], testX, testY)
print('Standard Accuracy: {:.4f}\n'.format(acc))

hypothesis = fit(trainX, trainY, 1, 1E-6)
print('Fitted Model: y = {1}x {2} {0}'.format(abs(hypothesis[0]), hypothesis[1], ('-', '+')[hypothesis[0]>0]))
acc = accuracy(hypothesis, testX, testY)
print('Fitted Accuracy: {:.4f}'.format(acc))
