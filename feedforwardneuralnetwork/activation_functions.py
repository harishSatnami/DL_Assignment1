# Activation functions
import math
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
    ans = 1/(1+math.exp(-a))
    return ans

def sigmoid_vector(a):
    for i in range(len(a)):
        a[i] = sigmoid(a[i])

# 3 Tanh
def tanh(a):
    ans = (math.exp(a)-math.exp(-a))/(math.exp(a)+math.exp(-a))
    return ans

def tanh_vector(a):
    for i in range(len(a)):
        a[i] = tanh(a[i])

# Output Activation Function

# Softmax
def softmax(a):
    summ = 0
    for i in a:
        summ+=math.exp(i)
    for i in range(len(a)):
        a[i] = math.exp(a[i])/summ
    return a