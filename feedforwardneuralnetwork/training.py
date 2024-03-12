


import math

from activation_functions import sigmoid_vector, relu_vector, tanh_vector
from utility import random_initialize, xavier_initialize

from gradient_descent_FFNN import gradient_descent_stochastic , gradient_descent_mini_batch, gradient_descent_momentum_based, gradient_descent_RMSProp, gradient_descent_nesterov_accelarated




def train_model(X, Y, epochs=1, num_of_hidden_layers=1, size_of_layers=4, learning_rate=0.1, optimizer="sgd", batch_size=4, l2_regularization_constant=0.001, weight_init_type="random", activation_function="sigmoid", beta=0, epsilon=1e-10):
    
    print("number of training datapoints:",X.shape[0])
    print("number of epochs:", epochs)
    print("number of hidden layers:", num_of_hidden_layers)
    print("size of hidden layers:", size_of_layers)
    print("learning rate:", learning_rate)
    print("optimizer:", optimizer)
    print("batch_size:", batch_size)
    print("l2 regularization constant:", l2_regularization_constant)
    print("weights and biases initialization type:", weight_init_type)
    print("activation function:", activation_function)
    print("beta:", beta)
    print("epsilon:", epsilon)



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
        Weights, Biases = gradient(X, Y, learning_rate, num_of_hidden_layers+2, batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta, epsilon)
        print("Epoch ",epoch+1, " finished.")
    return Weights, Biases