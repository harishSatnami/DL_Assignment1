from utility import *
from activation_functions import *

conf = {
    "epochs" : 5,
    "num_of_hidden_layer" : 3,
    "size_of_hidden_layer" : 32,
    "weight_decay" : 0,
    "learning_rate" : 1e-3,
    "optimizer" : "sgd",
    "batch_size" : 16,
    "weight_initializer" : random_initialize,
    "activation_function" : sigmoid_vector
}


'''number of epochs: 5, 10
number of hidden layers: 3, 4, 5
size of every hidden layer: 32, 64, 128
weight decay (L2 regularisation): 0, 0.0005, 0.5
learning rate: 1e-3, 1 e-4
optimizer: sgd, momentum, nesterov, rmsprop, adam, nadam
batch size: 16, 32, 64
weight initialisation: random, Xavier
activation functions: sigmoid, tanh, ReLU'''
