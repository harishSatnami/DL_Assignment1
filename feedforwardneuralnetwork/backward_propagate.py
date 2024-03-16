import numpy as np
from utility import hadamard_product
from derivatives import derivative_sigmoid, derivative_relu, derivative_tanh, derivative_identity


# Backward Propagation

def backward_propagation(H, A, W, y_actual, y_pred, number_of_layers, activation_function,loss_type):

    if activation_function=="relu":
        derivative = derivative_relu
    elif activation_function=="tanh":
        derivative = derivative_tanh
    elif activation_function=="identity":
        derivative = derivative_identity
    else:
        derivative = derivative_sigmoid
        
    delta_W = [0 for i in range(number_of_layers-1)]
    delta_B = [0 for i in range(number_of_layers-1)]

    # gradient with respect to output
    if loss_type=="mse" or loss_type=="mean_squared_error":
        delta_A  = 2 * (y_pred - y_actual) * y_pred * (1 - y_pred)
    else:
        delta_A = -(y_actual-y_pred)

    delta_H = None

    for k in reversed(range(number_of_layers-1)):

        # gradient with respect to parameters
        delta_W[k] = np.matmul(delta_A, H[k].transpose())
        delta_B[k] = np.sum(delta_A,axis=1)

        if k==0:
            break
        # gradient with respect to layer below
        delta_H = np.matmul(W[k].transpose() , delta_A)

        #gradient with respect to layer below (i.e. pre-activation)
        delta_A = hadamard_product(delta_H,[derivative(i) for i in A[k-1]])

    return delta_W, delta_B
