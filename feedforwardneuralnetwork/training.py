


import math

from activation_functions import sigmoid_vector, relu_vector, tanh_vector
from utility import random_initialize, xavier_initialize

from gradient_descent_FFNN import gradient_descent_stochastic , gradient_descent_mini_batch, gradient_descent_momentum_based, gradient_descent_RMSProp, gradient_descent_nesterov_accelarated




def train_model(X, Y, epochs=1, num_of_hidden_layers=1, size_of_layers=4, learning_rate=0.1, optimizer="sgd", batch_size=4, l2_regularization_constant=0.001, weight_init_type="random", activation_function="sigmoid", beta=0, epsilon=1e-10):

    if weight_init_type=="random":
        initialize = random_initialize
    else:
        initialize = xavier_initialize

    if optimizer=="mini_batch":
        gradient = gradient_descent_mini_batch
    elif optimizer=="mbgd":
        gradient = gradient_descent_momentum_based
    elif optimizer=="rmsprop":
        gradient = gradient_descent_RMSProp
    elif optimizer=="nagd":
        gradient = gradient_descent_nesterov_accelarated
    else:
        gradient = gradient_descent_stochastic

    Weights, Biases = initialize(num_of_hidden_layers+2,size_of_layers,Y.shape[1], X.shape[1])

    for epoch in range(epochs):
        print("Epoch number", epoch+1, " started")
        Weights, Biases = gradient(X, Y, learning_rate, num_of_hidden_layers+2, batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta=0, epsilon=0)
        print("Epoch ",epoch+1, " finished.")
    return Weights, Biases