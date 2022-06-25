from math import e
from numpy import where

def relu(x):
    x[where((x < 0))] = 0
    return x

def sigmoid(x):
    return 1 / (1 + e**(-x))

def tanh(x):
    return (e**(x) - e**(-x)) / (e**(x) + e**(-x))

def leakyRelu(x, a=0.2):
    return x if x > 0 else x * a
