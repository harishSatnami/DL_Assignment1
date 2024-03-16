# Activation functions
import numpy as np

# 1 Relu
def relu(a):
    return np.max(a,0)


def relu_vector(a):
    return np.maximum(a,0)


# 2 Sigmoid
def sigmoid(a):
    ans = 1/(1+np.exp(-a))
    return ans

def sigmoid_vector(a):
    a = np.clip(a, -200,200)
    ans = 1/(1+np.exp(-a))
    return ans

# 3 Tanh
def tanh(a):
    a = np.clip(a, -200, 200)
    ans = (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    return ans

def tanh_vector(a):
    return np.tanh(a)

# 4 Identity
def identity(a):
    return a


# Output Activation Function

# Softmax
def softmax(a):
    a = np.clip(a, -200, 200)
    return np.exp(a)/np.sum(np.exp(a))