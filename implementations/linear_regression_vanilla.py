# Univariate Linear Regression
# Simple, basic implementation using vanilla Python lists

import random
from math import sqrt

def gen_data(θ=None, θrange=(-10, 10, -20, 20), m=100, xMin=0, xMax=10, error=4):
    # Generate the correct answer for the data set
    if θ == None:
        if not (isinstance(θrange, tuple) or isinstance(θrange, list)) or \
            len(θrange) != 4 or \
            not all(isinstance(i, int) or isinstance(i, float) for i in θrange):
            return -1
        θ = (random.uniform(*θrange[:2]), random.uniform(*θrange[2:]))
    # Generate the data set
    data = []
    for i in range(m):
        x = random.uniform(xMin, xMax)
        y = θ[0] + θ[1] * x + random.uniform(-error, error)
        data.append((x, y))
    return θ, data

def J(θ, data, m):
    return sum(θ[0] + θ[1] * i[0] - i[1] for i in data) ** 2 / (2*m)

def partial_derivative(θ, variable, m):
    if variable == 0:
        return sum(θ[0] + θ[1] * i[0] - i[1] for i in data) / m
    elif variable == 1:
        return sum((θ[0] + θ[1] * i[0] - i[1]) * i[0] for i in data) / m

def gradient_descent(θ, α, m):
    newθ = []
    for j in range(len(θ)):
        newθ.append(θ[j] - α * partial_derivative(θ, j, m))
    return tuple(newθ)

def fit(data, learningRate=0.01, threashold=1E-10):
    m = len(data)
    θ = (0, 1)
    while max(abs(partial_derivative(θ, 0, m)), abs(partial_derivative(θ, 1, m))) > threashold:
        θ = gradient_descent(θ, learningRate, m)
    return θ

def accuracy(θ, testX, testY):
    # Returns the R^2 value of the model
    if len(testX) != len(testY):
        return -1
    explained = list(map(lambda x: θ[0] + θ[1] * x, testX))
    mean = sum(testY) / len(testY)
    totSumSquares = sum([(i - mean) ** 2 for i in testY])
    # regSumSquares = sum([(i - mean) ** 2 for i in explained])
    resSumSquares = sum([(testY[i] - explained[i]) ** 2 for i in range(len(testY))])
    accuracy = 1 - resSumSquares / totSumSquares
    return accuracy

dataSize = 200
answer, data = gen_data(m=dataSize)
trainX, trainY = [i[0] for i in data[:dataSize//2]], [i[1] for i in data[:dataSize//2]]
testX, testY = [i[0] for i in data[dataSize//2:]], [i[1] for i in data[dataSize//2:]]

print('Real Value: {}'.format(answer))
fitted = fit(data[:dataSize//2], 0.01, 1E-10)
print('Fitted Value: {}'.format(fitted))

acc = accuracy(fitted, testX, testY)
print('Accuracy: {:.4f}'.format(acc))
