from utility import random_initialize, get_average_delta_WandB
from forward_propagate import forward_propagation
from backward_propagate import backward_propagation

def update_weights_and_biases(learning_rate, Weights, Biases, delta_Weights, delta_Biases):
    for i in range(len(Weights)):
        # Weights[i] = Weights[i] - learning_rate * delta_Weights[i]
        # Biases[i] = Biases[i] - learning_rate * delta_Biases[i]
        for j in range(len(Weights[i])):
            Weights[i][j] = Weights[i][j] - learning_rate * delta_Weights[i][j]

        for j in range(len(Biases)):
            Biases[i][j] = Biases[i][j] - learning_rate * delta_Biases[i][j]

    return Weights, Biases



def gradient_descent(X, Y, learning_rate, number_of_layers, number_of_batch, batch_size, nodes_per_hidden_layer, nodes_in_output_layer, Weights, Biases):
    # Weights, Biases = random_initialize(number_of_layers,nodes_per_hidden_layer,nodes_in_output_layer)
    itr = 0

    while itr<X.shape[0]:
        # H, A, y_pred = forward_propagation(X[itr*batch_size:(itr+1)*batch_size], Weights, Biases, number_of_layers)
        H, A, y_pred = forward_propagation(X[itr], Weights, Biases, number_of_layers)
        # return None, None
        # delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size], y_pred, number_of_layers)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr], y_pred, number_of_layers)        
        Weights , Biases = update_weights_and_biases(learning_rate, Weights, Biases, delta_Weights, delta_Biases)
        itr = itr + 1
    return Weights, Biases


def gradient_descent_mini_batch(X, Y, learning_rate, number_of_layers, number_of_batchs, batch_size, size_of_hidden_layer, nodes_in_output_layer, Weights, Biases):
    itr = 0
    delta_W_acc = []
    delta_B_acc = []

    while itr<X.shape[0]:
        H, A, y_pred = forward_propagation(X[itr], Weights, Biases, number_of_layers)

        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr], y_pred, number_of_layers)

        delta_W_acc.append(delta_Weights)
        delta_B_acc.append(delta_Biases)

        itr = itr + 1
        if itr%batch_size==0:
            delta_W_avg, delta_B_avg = get_average_delta_WandB(delta_W_acc, delta_B_acc)

            Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, delta_W_avg, delta_B_avg)

            delta_W_acc = []
            delta_B_acc = []
            delta_W_avg = 0
            delta_B_avg = 0


    if delta_B_acc and delta_W_acc:
        delta_W_avg, delta_B_avg = get_average_delta_WandB(delta_W_acc, delta_B_acc)

        Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, delta_W_avg, delta_B_avg)

    return Weights, Biases

