import numpy as np
from utility import hadamard_product
from derivatives import derivative_sigmoid, derivative_relu, derivative_tanh


# Backward Propagation

def backward_propagation(H, A, W, y_actual, y_pred, number_of_layers, activation_function):
    
    if activation_function=="relu":
        derivative = derivative_relu
    elif activation_function=="tanh":
        derivative = derivative_tanh
    else:
        derivative = derivative_sigmoid
    # delta_A = [0 for i in range(number_of_layers-1)]
    delta_W = [0 for i in range(number_of_layers-1)]
    delta_B = [0 for i in range(number_of_layers-1)]
    # delta_H = [0 for i in range(number_of_layers-2)]

    # gradient with respect to output
    # delta_A[-1] = -(y_actual-y_pred)
    delta_A = -(y_actual-y_pred)
    delta_H = None


    for k in reversed(range(number_of_layers-1)):

        # gradient with respect to parameters
        # delta_W[k] = np.outer(delta_A[k],H[k-1])
        delta_W[k] = np.outer(delta_A, H[k])
        # delta_B[k] = delta_A[k]
        delta_B[k] = delta_A

        if k==0:
            break
        # gradient with respect to layer below
        # delta_H[k-1] = np.matmul(W[k].transpose() , delta_A[k])
        delta_H = np.matmul(W[k].transpose() , delta_A)

        #gradient with respect to layer below (i.e. pre-activation)
        # delta_A[k-1] = hadamard_product(delta_H[k-1],[derivative(i) for i in A[k-1]])
        delta_A = hadamard_product(delta_H,[derivative(i) for i in A[k-1]])
        

    return delta_W, delta_B