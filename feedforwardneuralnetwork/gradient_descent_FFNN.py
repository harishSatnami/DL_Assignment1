import numpy as np
from tqdm import tqdm
from utility import random_initialize, get_average_delta_WandB, hadamard_product
from forward_propagate import forward_propagation
from backward_propagate import backward_propagation


def update_weights_and_biases(learning_rate, Weights, Biases, delta_Weights, delta_Biases, l2_regularization_constant):
    for i in range(len(Weights)):
        # Weights[i] = Weights[i] - learning_rate * delta_Weights[i]
        # Biases[i] = Biases[i] - learning_rate * delta_Biases[i]
        for j in range(len(Weights[i])):
            Weights[i][j] = Weights[i][j] - learning_rate * delta_Weights[i][j] - (learning_rate * l2_regularization_constant * Weights[i][j])

        for j in range(len(Biases[i])):
            Biases[i][j] = Biases[i][j] - learning_rate * delta_Biases[i][j] #- (learning_rate * l2_regularization_constant * Biases[i][j])

    return Weights, Biases


def gradient_descent_stochastic(X, Y, learning_rate, number_of_layers, batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta=0, epsilon=0):

    # Weights, Biases = random_initialize(number_of_layers,nodes_per_hidden_layer,nodes_in_output_layer)
    # itr = 0

    for itr in tqdm(range(X.shape[0])):
        # H, A, y_pred = forward_propagation(X[itr*batch_size:(itr+1)*batch_size], Weights, Biases, number_of_layers)
        H, A, y_pred = forward_propagation(X[itr], Weights, Biases, number_of_layers, activation_function)
        # return None, None
        # delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size], y_pred, number_of_layers)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr], y_pred, number_of_layers, activation_function)        
        Weights , Biases = update_weights_and_biases(learning_rate, Weights, Biases, delta_Weights, delta_Biases, l2_regularization_constant)
        # itr = itr + 1
    return Weights, Biases

def gradient_descent_mini_batch(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta=0, epsilon=0):
    itr = 0
    delta_W_acc = []
    delta_B_acc = []

    for itr in tqdm(range(X.shape[0])):
        H, A, y_pred = forward_propagation(X[itr], Weights, Biases, number_of_layers, activation_function)

        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr], y_pred, number_of_layers, activation_function)

        delta_W_acc.append(delta_Weights)
        delta_B_acc.append(delta_Biases)

        # itr = itr + 1
        if (itr+1)%batch_size==0:
            delta_W_avg, delta_B_avg = get_average_delta_WandB(delta_W_acc, delta_B_acc)

            Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, delta_W_avg, delta_B_avg, l2_regularization_constant)

            delta_W_acc = []
            delta_B_acc = []
            delta_W_avg = 0
            delta_B_avg = 0


    if delta_B_acc and delta_W_acc:
        delta_W_avg, delta_B_avg = get_average_delta_WandB(delta_W_acc, delta_B_acc)

        Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, delta_W_avg, delta_B_avg, l2_regularization_constant)

    return Weights, Biases

def accumulate_history(prev, current, prev_factor=1, current_factor=1):
    temp = []
    for i in range(len(prev)):
        temp.append(np.add(prev[i]*prev_factor, current[i]*current_factor))

    return temp

def gradient_descent_momentum_based(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta, epsilon=0):
    
    itr = 0
    u_t_weights = [np.zeros_like(weight) for weight in Weights]
    u_t_biases = [np.zeros_like(bias) for bias in Biases]
    # u_t_list = [u_t]
    for itr in tqdm(range(X.shape[0])):
        H, A, y_pred = forward_propagation(X[itr], Weights, Biases, number_of_layers, activation_function)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr], y_pred, number_of_layers, activation_function)
        # u_t = beta * u_t + delta_Weights
        u_t_weights = accumulate_history(u_t_weights,delta_Weights,prev_factor=beta)
        u_t_biases = accumulate_history(u_t_biases,delta_Biases, prev_factor=beta)
        # u_t_list.append()
        # itr = itr + 1

        if (itr+1)%batch_size==0:
            Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, u_t_weights, u_t_biases, l2_regularization_constant)

    if itr%batch_size!=0:
        Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, u_t_weights, u_t_biases, l2_regularization_constant)

    return Weights, Biases


def square_each_term(a):
    temp = []
    for i in range(len(a)):
        temp.append(np.array(a[i])**2)
    return temp

def modify_deltas_RMSProp(v_t, w_t, epsilon):
    temp = []
    for i in range(len(v_t)):
        temp.append(np.divide(w_t[i], (np.sqrt(v_t[i]) + epsilon)))
    return temp

def gradient_descent_RMSProp(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta, epsilon):
    
    itr = 0
    v_t_weights = [np.zeros_like(weight) for weight in Weights]
    v_t_biases = [np.zeros_like(bias) for bias in Biases]

    for itr in tqdm(range(X.shape[0])):
        H, A, y_pred = forward_propagation(X[itr], Weights, Biases, number_of_layers, activation_function)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr], y_pred, number_of_layers, activation_function)

        v_t_weights = accumulate_history(v_t_weights,square_each_term(delta_Weights),prev_factor=beta, current_factor=1-beta)
        v_t_biases = accumulate_history(v_t_biases,square_each_term(delta_Biases), prev_factor=beta, current_factor=1-beta)

        # itr = itr + 1

        if (itr+1)%batch_size==0:
            Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, modify_deltas_RMSProp(v_t_weights, delta_Weights, epsilon), modify_deltas_RMSProp(v_t_biases, delta_Biases, epsilon),l2_regularization_constant)
    
    if itr%batch_size!=0:
        Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, modify_deltas_RMSProp(v_t_weights, delta_Weights, epsilon), modify_deltas_RMSProp(v_t_biases, delta_Biases, epsilon),l2_regularization_constant)
    
    
    return Weights, Biases


def modify_W_B_NAGD(u_t, w_t, beta):
    temp = []
    for i in range(len(u_t)):
        temp.append(np.subtract(w_t[i],beta*u_t[i]))
    return temp


def gradient_descent_nesterov_accelarated(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta, epsilon=0):
    itr = 0
    u_t_weights = [np.zeros_like(weight) for weight in Weights]
    u_t_biases = [np.zeros_like(bias) for bias in Biases]
    # u_t_list = [u_t]
    for itr in tqdm(range(X.shape[0])):
        H, A, y_pred = forward_propagation(X[itr], modify_W_B_NAGD(u_t_weights, Weights, beta), modify_W_B_NAGD(u_t_biases, Biases, beta), number_of_layers, activation_function)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr], y_pred, number_of_layers, activation_function)
        # u_t = beta * u_t + delta_Weights
        u_t_weights = accumulate_history(u_t_weights,delta_Weights,prev_factor=beta)
        u_t_biases = accumulate_history(u_t_biases,delta_Biases, prev_factor=beta)
        # u_t_list.append()
        # itr = itr + 1

        if (itr+1)%batch_size==0:
            Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, u_t_weights, u_t_biases, l2_regularization_constant)

    if itr%batch_size!=0:
        Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, u_t_weights, u_t_biases, l2_regularization_constant)


    return Weights, Biases