import numpy as np

def matrix_vector_multiply(M, V):
    return np.dot(M,V)

def hadamard_product(A,B):
    result = []
    for i in range(len(A)):
        result.append(A[i]*B[i])

    return result

def random_initialize(number_of_layers, nodes_per_hidden_layer, nodes_in_output_layer):
    W = np.random.rand(number_of_layers-1, nodes_per_hidden_layer ,nodes_per_hidden_layer)
    B = np.random.rand(number_of_layers-1, nodes_per_hidden_layer)
    WL = np.random.rand(nodes_per_hidden_layer,nodes_in_output_layer)
    BL = np.random.rand(nodes_in_output_layer)
    W_list = list(W).append(WL)
    B_list = list(B).append(BL)
    return W_list, B_list


def get_accuracy(Y_actual, Y_predicted):
    total = len(Y_actual)
    cnt = 0
    for i in range(total):
        if np.argmax(Y_actual[i]) == np.argmax(Y_predicted[i]):
            cnt = cnt + 1

    return (cnt/total)*100