# forward propagation 

import numpy as np

from pre_activation_functions import pre_activation
from activation_functions import sigmoid_vector, relu_vector, tanh_vector, identity, softmax



def forward_propagation(X, Weights, Biases, number_of_layers, activation_function, batch_size):

    if activation_function=="relu":
        activation = relu_vector
    elif activation_function=="tanh":
        activation = tanh_vector
    elif activation_function=="identity":
        activation = identity
    else:
        activation = sigmoid_vector

    A = []
    H = [X]
    for i in range(number_of_layers-2):

        modified_bias = Biases[i].reshape(1,-1)
        modified_bias_N = np.repeat(modified_bias, batch_size, axis=0).transpose()

        A.append(pre_activation(Weights[i],H[i],modified_bias_N))
        H.append(activation(A[i]))

    modified_bias = Biases[-1].reshape(1,-1)
    modified_bias_N = np.repeat(modified_bias, batch_size, axis=0).transpose()

    A.append(pre_activation(Weights[-1], H[-1], modified_bias_N))

    y_pred_temp = []
    A_trns = A[-1].transpose()
    for i in range(batch_size):
        y_pred_temp.append(softmax(A_trns[i]))

    y_pred = np.array(y_pred_temp).transpose()

    return H, A, y_pred
