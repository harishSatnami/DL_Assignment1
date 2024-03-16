import numpy as np
from activation_functions import sigmoid, tanh

# Derivative functions

def derivative_sigmoid(a):
    return sigmoid(a) * (1-sigmoid(a))

def derivative_tanh(a):
    return 1 - (tanh(a)**2)

def derivative_relu(a):
    a[a<=0] = 0
    a[a>0] = 1
    return a

def derivative_identity(a):
    return np.ones_like(a)

