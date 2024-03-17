import numpy as np


def random_initialize(number_of_layers, nodes_per_hidden_layer, nodes_in_output_layer, input_layer_size, batch_size):
    if number_of_layers<=2:
        return [np.random.randn(nodes_in_output_layer,input_layer_size)], [np.random.randn(nodes_in_output_layer)]

    if number_of_layers==3:
        Weights = [np.random.randn(nodes_per_hidden_layer, input_layer_size), np.random.randn(nodes_in_output_layer, nodes_per_hidden_layer)]
        Biases = [np.random.randn(nodes_per_hidden_layer), np.random.randn(nodes_in_output_layer)]
        return Weights, Biases

    WS = np.random.randn(nodes_per_hidden_layer, input_layer_size)
    W = np.random.randn(number_of_layers-3, nodes_per_hidden_layer ,nodes_per_hidden_layer)
    B = np.random.randn(number_of_layers-2, 1, nodes_per_hidden_layer)
    WL = np.random.randn(nodes_in_output_layer, nodes_per_hidden_layer)
    BL = np.random.randn(1, 1, nodes_in_output_layer)


    

    # BL = np.repeat(BL,batch_size,axis=1)

    Weights = [WS] + [i for i in W] + [WL]
    Biases = [i for i in B] + [i for i in BL]
    
    for i in range(len(Biases)):
        Biases[i] = np.repeat(Biases[i].transpose(),batch_size, axis=1)
    
    return Weights, Biases


from pre_activation_functions import pre_activation
from activation_functions import sigmoid_vector, relu_vector, tanh_vector, softmax


def forward_propagation_n(X, Weights, Biases, number_of_layers, activation_function, batch_size):
    
    if activation_function=="relu":
        activation = relu_vector
    elif activation_function=="tanh":
        activation = tanh_vector
    else:
        activation = sigmoid_vector
    
    A = []
    H = [X]
    for i in range(number_of_layers-2):
        
        modified_bias = Biases[i].reshape(1,-1)
        modified_bias_N = np.repeat(modified_bias, batch_size, axis=0)
        
        A.append(pre_activation(Weights[i],H[i],modified_bias_N))
        H.append(activation(A[i]))

    modified_bias = Biases[-1].reshape(1,-1)
    modified_bias_N = np.repeat(modified_bias, batch_size, axis=0)

    A.append(pre_activation(Weights[-1], H[-1], modified_bias_N))

    y_pred_temp = []
    A_trns = A[-1].transpose()
    for i in range(X.shape[0]):
        y_pred_temp.append(softmax(A_trns[i]))

    y_pred = np.array(y_pred_temp).transpose()

    return H, A, y_pred


from utility import hadamard_product
from derivatives import derivative_sigmoid, derivative_relu, derivative_tanh


# Backward Propagation

def backward_propagation_n(H, A, W, y_actual, y_pred, number_of_layers, activation_function):
    
    if activation_function=="relu":
        derivative = derivative_relu
    elif activation_function=="tanh":
        derivative = derivative_tanh
    else:
        derivative = derivative_sigmoid
    # delta_A = [0 for i in range(number_of_layers-1)]
    delta_W = [0 for i in range(number_of_layers-1)]
    delta_B = [0 for i in range(number_of_layers-1)]
    # delta_H = [0 for i in range(number_of_layers-2)]

    # gradient with respect to output
    # delta_A[-1] = -(y_actual-y_pred)
    delta_A = -(y_actual-y_pred)
    delta_H = None


    for k in reversed(range(number_of_layers-1)):

        # gradient with respect to parameters
        # delta_W[k] = np.outer(delta_A[k],H[k])
        delta_W[k] = np.matmul(delta_A, H[k].transpose())
        # delta_B[k] = delta_A[k]
        delta_B[k] = delta_A

        if k==0:
            break
        # gradient with respect to layer below
        # delta_H[k-1] = np.matmul(W[k].transpose() , delta_A[k])
        delta_H = np.matmul(W[k].transpose() , delta_A)

        #gradient with respect to layer below (i.e. pre-activation)
        # delta_A[k-1] = hadamard_product(delta_H[k-1],[derivative(i) for i in A[k-1]])
        delta_A = hadamard_product(delta_H,[derivative(i) for i in A[k-1]])
        
        delta_B[k] = np.sum(delta_B[k],axis=0)
            

    return delta_W, delta_B






#------------------------------------------final-----------------------------------


# new optimization


def forward_propagation_n(X, Weights, Biases, number_of_layers, activation_function, batch_size):
    
    if activation_function=="relu":
        activation = relu_vector
    elif activation_function=="tanh":
        activation = tanh_vector
    else:
        activation = sigmoid_vector
    
    A = []
    H = [X]
    for i in range(number_of_layers-2):
        
        modified_bias = Biases[i].reshape(1,-1)
        modified_bias_N = np.repeat(modified_bias, batch_size, axis=0).transpose()
        
        A.append(pre_activation(Weights[i],H[i],modified_bias_N))
        H.append(activation(A[i]))

    modified_bias = Biases[-1].reshape(1,-1)
    modified_bias_N = np.repeat(modified_bias, batch_size, axis=0).transpose()

    A.append(pre_activation(Weights[-1], H[-1], modified_bias_N))

    y_pred_temp = []
    A_trns = A[-1].transpose()
    for i in range(batch_size):
        y_pred_temp.append(softmax(A_trns[i]))

    y_pred = np.array(y_pred_temp).transpose()

    return H, A, y_pred


# Backward Propagation

def backward_propagation_n(H, A, W, y_actual, y_pred, number_of_layers, activation_function):
    
    if activation_function=="relu":
        derivative = derivative_relu
    elif activation_function=="tanh":
        derivative = derivative_tanh
    else:
        derivative = derivative_sigmoid
    # delta_A = [0 for i in range(number_of_layers-1)]
    delta_W = [0 for i in range(number_of_layers-1)]
    delta_B = [0 for i in range(number_of_layers-1)]
    # delta_H = [0 for i in range(number_of_layers-2)]

    # gradient with respect to output
    # delta_A[-1] = -(y_actual-y_pred)
    delta_A = -(y_actual-y_pred)
    delta_H = None


    for k in reversed(range(number_of_layers-1)):

        # gradient with respect to parameters
        # delta_W[k] = np.outer(delta_A[k],H[k])
        delta_W[k] = np.matmul(delta_A, H[k].transpose())
        # delta_B[k] = delta_A[k]
        delta_B[k] = np.sum(delta_A,axis=1)

        if k==0:
            break
        # gradient with respect to layer below
        # delta_H[k-1] = np.matmul(W[k].transpose() , delta_A[k])
        delta_H = np.matmul(W[k].transpose() , delta_A)

        #gradient with respect to layer below (i.e. pre-activation)
        # delta_A[k-1] = hadamard_product(delta_H[k-1],[derivative(i) for i in A[k-1]])
        delta_A = hadamard_product(delta_H,[derivative(i) for i in A[k-1]])
        
        # delta_B[k] = np.sum(delta_B[k],axis=0)
            

    return delta_W, delta_B




#---------------------------------------------------------------


# gradient descent algorithms


def update_weights_and_biases_n(learning_rate, Weights, Biases, delta_Weights, delta_Biases, l2_regularization_constant):
    
    for i in range(len(Weights)):
        # print("+++++++++++++++++++++++++++")
        # print(Biases[i].shape)
        # print(delta_Biases[i].shape)
        # Weights[i] = Weights[i] - learning_rate * delta_Weights[i]
        # Biases[i] = Biases[i] - learning_rate * delta_Biases[i]
        for j in range(len(Weights[i])):
            Weights[i][j] = Weights[i][j] - learning_rate * delta_Weights[i][j] - (learning_rate * l2_regularization_constant * Weights[i][j])

        for j in range(len(Biases[i])):
            Biases[i][j] = Biases[i][j] - learning_rate * delta_Biases[i][j] #- (learning_rate * l2_regularization_constant * Biases[i][j])

    return Weights, Biases



def gradient_descent_mini_batch_n(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta=0, epsilon=0):
    itr = 0
    # delta_W_acc = []
    # delta_B_acc = []

    for itr in tqdm(range(X.shape[0]//batch_size)):
        H, A, y_pred = forward_propagation_n(X[itr*batch_size:(itr+1)*batch_size].transpose(), Weights, Biases, number_of_layers, activation_function, batch_size)

        delta_Weights, delta_Biases = backward_propagation_n(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size].transpose(), y_pred, number_of_layers, activation_function)

        # delta_W_acc.append(delta_Weights)
        # delta_B_acc.append(delta_Biases)

        # itr = itr + 1
        # if (itr+1)%batch_size==0:
        # delta_W_avg, delta_B_avg = get_average_delta_WandB(delta_W_acc, delta_B_acc)

        Weights, Biases = update_weights_and_biases_n(learning_rate, Weights, Biases, delta_Weights, delta_Biases, l2_regularization_constant)

        # delta_W_acc = []
        # delta_B_acc = []
        # delta_W_avg = 0
        # delta_B_avg = 0

    # if X.shape[0]-itr

    # if delta_B_acc and delta_W_acc:
    #     delta_W_avg, delta_B_avg = get_average_delta_WandB(delta_W_acc, delta_B_acc)

        # Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, delta_W_avg, delta_B_avg, l2_regularization_constant)

    return Weights, Biases


def accumulate_history(prev, current, prev_factor=1, current_factor=1):
    temp = []
    for i in range(len(prev)):
        temp.append(np.add(prev[i]*prev_factor, current[i]*current_factor))

    return temp

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
        H, A, y_pred = forward_propagation_n(X[itr*batch_size:(itr+1)*batch_size].transpose(), modify_W_B_NAGD(u_t_weights, Weights, beta), modify_W_B_NAGD(u_t_biases, Biases, beta), number_of_layers, activation_function)
        delta_Weights, delta_Biases = backward_propagation_n(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size].transpose(), y_pred, number_of_layers, activation_function)
        # u_t = beta * u_t + delta_Weights
        u_t_weights = accumulate_history(u_t_weights,delta_Weights,prev_factor=beta)
        u_t_biases = accumulate_history(u_t_biases,delta_Biases, prev_factor=beta)
        # u_t_list.append()
        # itr = itr + 1

        # if (itr+1)%batch_size==0:
        Weights, Biases = update_weights_and_biases_n(learning_rate, Weights, Biases, u_t_weights, u_t_biases, l2_regularization_constant)

    # if itr%batch_size!=0:
    #     Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, u_t_weights, u_t_biases, l2_regularization_constant)


    return Weights, Biases