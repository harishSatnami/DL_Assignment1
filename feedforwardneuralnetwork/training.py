
'''number of epochs: 5, 10
number of hidden layers: 3, 4, 5
size of every hidden layer: 32, 64, 128
weight decay (L2 regularisation): 0, 0.0005, 0.5
learning rate: 1e-3, 1 e-4
optimizer: sgd, momentum, nesterov, rmsprop, adam, nadam
batch size: 16, 32, 64
weight initialisation: random, Xavier
activation functions: sigmoid, tanh, ReLU'''


import math

from activation_functions import sigmoid_vector, relu_vector, tanh_vector
from utility import random_initialize

def train_model(X, Y, epochs=1, num_of_hidden_layers=1, size_of_layers=4, learning_rate=0.1, optimizer="sgd", batch_size=4, weight_init_type=random_initialize, activation_function=sigmoid_vector):

    num_of_datapoints = X.shape[0]
    num_of_batchs = math.ceil(num_of_datapoints / batch_size)
    Weights, Biases = random_initialize(num_of_hidden_layers+1,size_of_layers,Y.shape[1])

    for epoch in epochs:
        pass

    pass