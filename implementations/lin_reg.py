# Multivariate Linear Regression
# Includes data normalization

import numpy as np
from sklearn import datasets, linear_model

def partial_derivative(theta, trainX, trainY, trainSize):
    return trainX.T.dot((trainX.dot(theta) - trainY)) / trainSize

def normalize(data):
    mean = np.mean(data, axis=0)
    stdDeviation = np.std(data, axis=0)
    return (data - mean) / stdDeviation

def fit(trainX, trainY, learningRate=0.1, threashold=1E-5):
    trainSize, featureSize = trainX.shape
    theta = np.array(np.ones((featureSize, 1)), dtype=float) # Initialize theta
    derivative = partial_derivative(theta, trainX, trainY, trainSize)
    while abs(derivative).max() > threashold: # While gradient doesn't converge
        theta -= learningRate * derivative # Perform gradient descent
        derivative = partial_derivative(theta, trainX, trainY, trainSize) # Update partial derivative
    return theta

def standard_fit(trainX, trainY):
    trainX = trainX[:, 1:] # Remove the extra bias term
    model = linear_model.LinearRegression()
    model.fit(trainX, trainY)
    return np.append(model.intercept_, model.coef_)[np.newaxis].T # Returns the hypothesis in a column vector

def accuracy(theta, testX, testY):
    # Returns the R Squared value of the model
    explained = testX.dot(theta)
    mean = np.sum(testY) / len(testY)
    totSumSquares = np.sum((testY - mean) ** 2)
    resSumSquares = np.sum((testY - explained) ** 2)
    accuracy = 1 - resSumSquares / totSumSquares
    return accuracy

dataset = datasets.load_diabetes()
data, target = dataset.data, dataset.target
dataSize = data.shape[0] # Number of rows (entries)
data = normalize(data) # Normalize the data
data = np.hstack((np.ones((dataSize, 1)), data)) # Add an extra bias term
target = target[np.newaxis].T # Transpose target vector into column vector
separation = int(dataSize * 0.8) # Percentage of training data set and testing data set
trainX, trainY = data[:separation, :], target[:separation]
testX, testY = data[separation:, :], target[separation:]

hypothesis = standard_fit(trainX, trainY)
print('Standard Fitted Model: {}'.format(hypothesis.T))
acc = accuracy(hypothesis, testX, testY)
print('Standard Fitted Accuracy: {:.4f}\n'.format(acc))

hypothesis = fit(trainX, trainY, 0.1, 1E-6)
print('Fitted Model: {}'.format(hypothesis.T))
acc = accuracy(hypothesis, testX, testY)
print('Fitted Accuracy: {:.4f}'.format(acc))
