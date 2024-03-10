


import math

from activation_functions import sigmoid_vector, relu_vector, tanh_vector
from utility import random_initialize
from gradient_descent_FFNN import gradient_descent

def train_model(X, Y, epochs=1, num_of_hidden_layers=1, size_of_layers=4, learning_rate=0.1, optimizer="sgd", batch_size=4, weight_init_type=random_initialize, activation_function=sigmoid_vector):

    num_of_datapoints = X.shape[0]
    num_of_batchs = math.ceil(num_of_datapoints / batch_size)
    Weights, Biases = weight_init_type(num_of_hidden_layers+2,size_of_layers,Y.shape[1], X.shape[1])

    for epoch in range(epochs):
        print("Epoch number", epoch+1, " started")
        Weights, Biases = gradient_descent(X,Y,learning_rate,num_of_hidden_layers+2,num_of_batchs,batch_size,size_of_layers,Y.shape[1],Weights,Biases)
        print("Epoch ",epoch, " finished.")
    return Weights, Biases