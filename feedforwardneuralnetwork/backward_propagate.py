import numpy as np
from utility import hadamard_product

# Backward Propagation

def backward_propagation(H, A, W, y_actual, y_pred, number_of_layers):
    delta_A = [0 for i in range(number_of_layers)]
    delta_W = [0 for i in range(number_of_layers)]
    delta_B = [0 for i in range(number_of_layers)]
    delta_H = [0 for i in range(number_of_layers-1)]

    # gradient with respect to output
    delta_A[-1] = -(y_actual-y_pred)

    for k in reversed(range(number_of_layers)):

        # gradient with respect to parameters
        delta_W[k] = np.outer(delta_A[k],H[k-1])
        delta_B[k] = delta_A[k]

        if k==0:
            break
        # gradient with respect to layer below
        delta_H[k-1] = np.matmul(W[k].tranpose() , delta_A[k])

        #gradient with respect to layer below (i.e. pre-activation)
        delta_A[k-1] = hadamard_product(delta_H[k-1],[derivative(i) for i in A[k-1]])
        

    return delta_W, delta_B