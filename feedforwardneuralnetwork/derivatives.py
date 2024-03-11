from activation_functions import sigmoid, relu, tanh

def derivative_sigmoid(a):
    return sigmoid(a) * (1-sigmoid(a))

def derivative_tanh(a):
    return 1 - (tanh(a)**2)

def derivative_relu(a):
    if a<=0:
        return 0
    return 1

