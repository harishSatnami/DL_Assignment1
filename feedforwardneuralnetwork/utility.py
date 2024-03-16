# All utility functions

import numpy as np

def hadamard_product(A,B):
    return np.multiply(np.array(A),np.array(B))


def random_initialize(number_of_layers, nodes_per_hidden_layer, nodes_in_output_layer, input_layer_size=784):
    if number_of_layers<=2:
        return [np.random.randn(nodes_in_output_layer,input_layer_size)], [np.random.randn(nodes_in_output_layer)]

    if number_of_layers==3:
        Weights = [np.random.randn(nodes_per_hidden_layer, input_layer_size), np.random.randn(nodes_in_output_layer, nodes_per_hidden_layer)]
        Biases = [np.random.randn(nodes_per_hidden_layer), np.random.randn(nodes_in_output_layer)]
        return Weights, Biases

    WS = np.random.randn(nodes_per_hidden_layer, input_layer_size)*0.1
    W = np.random.randn(number_of_layers-3, nodes_per_hidden_layer ,nodes_per_hidden_layer)*0.1
    B = np.random.randn(number_of_layers-2, nodes_per_hidden_layer)*0.1
    WL = np.random.randn(nodes_in_output_layer, nodes_per_hidden_layer)*0.1
    BL = np.random.randn(nodes_in_output_layer)*0.1

    Weights = [WS] + [i for i in W] + [WL]
    Biases = [i for i in B] + [BL]
    return Weights, Biases

def xavier_initialize(number_of_layers, nodes_per_hidden_layer, nodes_in_output_layer, input_layer_size=784):
    if number_of_layers<=2:
        return [np.random.randn(nodes_in_output_layer,input_layer_size)], [np.random.randn(nodes_in_output_layer)]

    fact_in = np.sqrt(6/(input_layer_size + nodes_per_hidden_layer))
    fact_out = np.sqrt(6/(nodes_in_output_layer + nodes_per_hidden_layer))
    fact_hid = np.sqrt(6/(nodes_per_hidden_layer + nodes_per_hidden_layer))

    if number_of_layers==3:
        Weights = [np.random.uniform(-fact_in, fact_in, (nodes_per_hidden_layer, input_layer_size)), np.random.uniform(-fact_out,fact_out,(nodes_in_output_layer, nodes_per_hidden_layer))]
        Biases = [np.zeros(nodes_per_hidden_layer), np.zeros(nodes_in_output_layer)]
        return Weights, Biases

    WS = np.random.uniform(-fact_in, fact_in, (nodes_per_hidden_layer, input_layer_size))
    W = np.random.uniform(-fact_hid, fact_hid, (number_of_layers-3, nodes_per_hidden_layer ,nodes_per_hidden_layer))
    B = np.zeros([number_of_layers-2, nodes_per_hidden_layer])
    WL = np.random.uniform(-fact_out,fact_out,(nodes_in_output_layer, nodes_per_hidden_layer))
    BL = np.zeros(nodes_in_output_layer)

    Weights = [WS] + [i for i in W] + [WL]
    Biases = [i for i in B] + [BL]
    return Weights, Biases




def get_accuracy(Y_actual, Y_predicted):
    total = len(Y_actual)
    print("total data points in validation ",total)
    cnt = 0
    for i in range(total):
        if np.argmax(Y_actual[i]) == np.argmax(Y_predicted[i]):
            cnt = cnt + 1

    return (cnt/total)*100


def get_average_delta_WandB(delta_W_acc, delta_B_acc):
    for i in range(1,len(delta_W_acc)):
        for j in range(len(delta_W_acc[0])):
            delta_W_acc[0][j] = np.add(delta_W_acc[0][j] , delta_W_acc[i][j])

        for j in range(len(delta_B_acc[0])):
            delta_B_acc[0][j] = np.add(delta_B_acc[0][j] , delta_B_acc[i][j])
            
    return delta_W_acc[0], delta_B_acc[0]

# loss

def cross_entropy_loss( y_actual, y_pred):
    loss = 0
    for i in range(len(y_actual)):
        loss = loss + (-np.log(y_pred[i][np.argmax(y_actual[i])] + 1e-10))
    return loss/len(y_actual)

def mean_squared_error( y_actual, y_pred):
    loss = 0
    for i in range(len(y_actual)):
        for j in range(len(y_actual[i])):
            loss = loss + (y_actual[i][j]-y_pred[i][j])**2
    loss = loss / (len(y_actual)*len(y_actual[0]))
    return loss