# Activation functions
import math
import numpy as np
# 1 Relu
def relu(a):
    if a < 0 :
        return 0
    return a

def relu_vector(a):
    temp = []
    for i in range(len(a)):
        temp.append(relu(a[i]))
    return temp

# 2 Sigmoid
def sigmoid(a):
    ans = 1/(1+np.exp(-a))
    return ans

def sigmoid_vector_old(a):
    temp=[]
    for i in range(len(a)):
        temp.append(sigmoid(a[i]))
    return temp

def sigmoid_vector(a):
    a = np.clip(a, -200,200)
    ans = 1/(1+np.exp(-a))
    return ans

# 3 Tanh
def tanh(a):
    a = np.clip(a, -200, 200)
    ans = (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    return ans

def tanh_vector_old(a):
    temp = []
    for i in range(len(a)):
        temp.append(tanh(a[i]))
    return temp

def tanh_vector(a):
    a = np.clip(a, -200, 200)
    ans = (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    return ans

# Output Activation Function

# Softmax
def softmax(a):
    a = np.clip(a, -200, 200)
    return np.exp(a)/np.sum(np.exp(a))