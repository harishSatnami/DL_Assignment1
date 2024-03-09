import numpy as np

def hadamard_product(A,B):
    result = []
    for i in range(len(A)):
        result.append(A[i]*B[i])

    return result

def random_initialize(number_of_layers, nodes_per_hidden_layer, nodes_in_output_layer, input_layer_size=784):
    if number_of_layers<=2:
        return [np.random.rand(nodes_in_output_layer,input_layer_size)], [np.random.rand(nodes_in_output_layer)]

    if number_of_layers==3:
        Weights = [np.random.rand(nodes_per_hidden_layer, input_layer_size), np.random.rand(nodes_in_output_layer, nodes_per_hidden_layer)]
        Biases = [np.random.rand(nodes_per_hidden_layer), np.random.rand(nodes_in_output_layer)]
        return Weights, Biases

    WS = np.random.rand(nodes_per_hidden_layer, input_layer_size)
    W = np.random.rand(number_of_layers-2, nodes_per_hidden_layer ,nodes_per_hidden_layer)
    B = np.random.rand(number_of_layers-1, nodes_per_hidden_layer)
    WL = np.random.rand(nodes_in_output_layer, nodes_per_hidden_layer)
    BL = np.random.rand(nodes_in_output_layer)

    Weights = [WS] + [i for i in W] + [WL]
    Biases = [i for i in B] + [BL]
    return Weights, Biases


def get_accuracy(Y_actual, Y_predicted):
    total = len(Y_actual)
    cnt = 0
    for i in range(total):
        if np.argmax(Y_actual[i]) == np.argmax(Y_predicted[i]):
            cnt = cnt + 1

    return (cnt/total)*100