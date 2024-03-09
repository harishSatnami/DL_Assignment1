from utility import random_initialize
from forward_propagate import forward_propagation
from backward_propagate import backward_propagation

def update_weights_and_biases(learning_rate, Weights, Biases, delta_Weights, delta_Biases):
    for i in range(len(Weights)):
        Weights[i] = Weights[i] - learning_rate * delta_Weights[i]
        Biases[i] = Biases[i] - learning_rate * delta_Biases[i]

    return Weights, Biases



def gradient_descent(X, Y, learning_rate, number_of_layers, number_of_batch, batch_size, nodes_per_hidden_layer, nodes_in_output_layer, Weights, Biases):
    # Weights, Biases = random_initialize(number_of_layers,nodes_per_hidden_layer,nodes_in_output_layer)
    itr = 0
    while itr<number_of_batch:
        H, A, y_pred = forward_propagation(X[itr*batch_size:(itr+1)*batch_size], Weights, Biases, number_of_layers)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size], y_pred, number_of_layers)
        Weights , Biases = update_weights_and_biases(learning_rate, Weights, Biases, delta_Weights, delta_Biases)
    
    return Weights, Biases