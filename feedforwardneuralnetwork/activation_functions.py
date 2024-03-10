# Activation functions
import math
import numpy as np
# 1 Relu
def relu(a):
    if a < 0 :
        return 0
    return a

def relu_vector(a):
    for i in range(len(a)):
        a[i] = relu(a[i])
    return a

# 2 Sigmoid
def sigmoid(a):
    ans = 1/(1+np.exp(-a))
    return ans

def sigmoid_vector_old(a):
    for i in range(len(a)):
        a[i] = sigmoid(a[i])
    return a

def sigmoid_vector(a):
    a = np.clip(a, -200,200)
    ans = 1/(1+np.exp(-a))
    return ans

# 3 Tanh
def tanh(a):
    ans = (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    return ans

def tanh_vector(a):
    for i in range(len(a)):
        a[i] = tanh(a[i])
    return a

# Output Activation Function

# Softmax
def softmax(a):
    a = np.clip(a, -200, 200)
    # summ = 0
    # for i in a:
    #     summ+=np.exp(i)
    # for i in range(len(a)):
    #     a[i] = np.exp(a[i])/summ
    return np.exp(a)/np.sum(np.exp(a))