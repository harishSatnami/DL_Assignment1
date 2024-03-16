import numpy as np
from tqdm import tqdm
from forward_propagate import forward_propagation
from backward_propagate import backward_propagation



def update_weights_and_biases(learning_rate, Weights, Biases, delta_Weights, delta_Biases, l2_regularization_constant):

    for i in range(len(Weights)):

        for j in range(len(Weights[i])):
            Weights[i][j] = Weights[i][j] - learning_rate * delta_Weights[i][j] - (learning_rate * l2_regularization_constant * Weights[i][j])

        for j in range(len(Biases[i])):
            Biases[i][j] = Biases[i][j] - learning_rate * delta_Biases[i][j] #- (learning_rate * l2_regularization_constant * Biases[i][j])

    return Weights, Biases


def gradient_descent_mini_batch(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, momentum=0, beta=0, beta1=0, beta2=0, epsilon=0, loss_type="cross_entropy"):
    itr = 0
    
    for itr in tqdm(range(X.shape[0]//batch_size)):
        H, A, y_pred = forward_propagation(X[itr*batch_size:(itr+1)*batch_size].transpose(), Weights, Biases, number_of_layers, activation_function, batch_size)

        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size].transpose(), y_pred, number_of_layers, activation_function, loss_type)

        Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, delta_Weights, delta_Biases, l2_regularization_constant)

    return Weights, Biases



def accumulate_history(prev, current, prev_factor=1, current_factor=1):
    temp = []
    for i in range(len(prev)):
        temp.append((prev[i]*prev_factor) + (current[i]*current_factor))

    return temp


def gradient_descent_momentum_based(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, momentum, beta=0, beta1=0, beta2=0, epsilon=0, loss_type="cross_entropy"):

    itr = 0
    u_t_weights = [np.zeros_like(weight) for weight in Weights]
    u_t_biases = [np.zeros_like(bias) for bias in Biases]
    for itr in tqdm(range(X.shape[0]//batch_size)):
        H, A, y_pred = forward_propagation(X[itr*batch_size:(itr+1)*batch_size].transpose(), Weights, Biases, number_of_layers, activation_function,batch_size)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size].transpose(), y_pred, number_of_layers, activation_function, loss_type)
        u_t_weights = accumulate_history(u_t_weights,delta_Weights,prev_factor=momentum)
        u_t_biases = accumulate_history(u_t_biases,delta_Biases, prev_factor=momentum)

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
        temp.append(w_t[i] / (np.sqrt(v_t[i]) + epsilon))
    return temp


def gradient_descent_RMSProp(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta, momentum=0, beta1=0, beta2=0, epsilon=0, loss_type="cross_entropy"):

    itr = 0
    v_t_weights = [np.zeros_like(weight) for weight in Weights]
    v_t_biases = [np.zeros_like(bias) for bias in Biases]

    for itr in tqdm(range(X.shape[0]//batch_size)):

        H, A, y_pred = forward_propagation(X[itr*batch_size:(itr+1)*batch_size].transpose(), Weights, Biases, number_of_layers, activation_function,batch_size)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size].transpose(), y_pred, number_of_layers, activation_function, loss_type)

        v_t_weights = accumulate_history(v_t_weights,square_each_term(delta_Weights),prev_factor=beta, current_factor=1-beta)
        v_t_biases = accumulate_history(v_t_biases,square_each_term(delta_Biases), prev_factor=beta, current_factor=1-beta)

        Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, modify_deltas_RMSProp(v_t_weights, delta_Weights, epsilon), modify_deltas_RMSProp(v_t_biases, delta_Biases, epsilon),l2_regularization_constant)

    return Weights, Biases


def modify_W_B_NAGD(u_t, w_t, beta):
    temp = []
    for i in range(len(u_t)):
        temp.append(w_t[i]- (beta*u_t[i]))
    return temp


def gradient_descent_nesterov_accelarated(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, momentum, beta=0, beta1=0, beta2=0, epsilon=0, loss_type="cross_entropy"):
    itr = 0
    g_t_weights = [np.zeros_like(weight) for weight in Weights]
    g_t_biases = [np.zeros_like(bias) for bias in Biases]
    m_t_weights = [np.zeros_like(weight) for weight in Weights]
    m_t_biases = [np.zeros_like(bias) for bias in Biases]

    for itr in tqdm(range(X.shape[0]//batch_size)):
        H, A, y_pred = forward_propagation(X[itr*batch_size:(itr+1)*batch_size].transpose(), Weights, Biases, number_of_layers, activation_function, batch_size)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size].transpose(), y_pred, number_of_layers, activation_function, loss_type)

        g_t_weights = delta_Weights
        g_t_biases = delta_Biases

        m_t_weights = accumulate_history(m_t_weights,g_t_weights,prev_factor=momentum)
        m_t_biases = accumulate_history(m_t_biases,g_t_biases, prev_factor=momentum)

        u_t_weights = accumulate_history(m_t_weights,g_t_weights,prev_factor=momentum)
        u_t_biases = accumulate_history(m_t_biases,g_t_biases, prev_factor=momentum)

        Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, u_t_weights, u_t_biases, l2_regularization_constant)

    return Weights, Biases


def update_theta_hat_adam(theta,beta,itr):
    temp = []
    for i in theta:
        temp.append(i/(1-np.power(beta,itr+1)))

    return temp

def modify_deltas_adam(m_theta_hat, v_theta_hat, epsilon):
    temp = []
    for  i in range(len(m_theta_hat)):
        temp.append(m_theta_hat[i]/(np.sqrt(v_theta_hat[i])+epsilon))

    return temp


def gradient_descent_adam(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta1, beta2, epsilon, momentum=0, beta=0, loss_type="cross_entropy"):

    itr = 0
    v_t_weights = [np.zeros_like(weight) for weight in Weights]
    v_t_biases = [np.zeros_like(bias) for bias in Biases]
    m_weights = [np.zeros_like(weight) for weight in Weights]
    m_biases = [np.zeros_like(bias) for bias in Biases]
    m_w_hat = [np.zeros_like(weight) for weight in Weights]
    m_b_hat = [np.zeros_like(bias) for bias in Biases]
    v_w_hat = [np.zeros_like(weight) for weight in Weights]
    v_b_hat = [np.zeros_like(bias) for bias in Biases]

    for itr in tqdm(range(X.shape[0]//batch_size)):

        H, A, y_pred = forward_propagation(X[itr*batch_size:(itr+1)*batch_size].transpose(), Weights, Biases, number_of_layers, activation_function,batch_size)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size].transpose(), y_pred, number_of_layers, activation_function, loss_type)


        m_weights = accumulate_history(m_weights,delta_Weights,prev_factor=beta1, current_factor=1-beta1)
        m_biases = accumulate_history(m_biases,delta_Biases, prev_factor=beta1, current_factor=1-beta1)

        v_t_weights = accumulate_history(v_t_weights,square_each_term(delta_Weights),prev_factor=beta2, current_factor=1-beta2)
        v_t_biases = accumulate_history(v_t_biases,square_each_term(delta_Biases), prev_factor=beta2, current_factor=1-beta2)

        m_w_hat = update_theta_hat_adam(m_weights,beta1,itr)
        m_b_hat = update_theta_hat_adam(m_biases,beta1,itr)
        
        v_w_hat = update_theta_hat_adam(v_t_weights,beta2,itr)
        v_b_hat = update_theta_hat_adam(v_t_biases,beta2,itr)

        Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, modify_deltas_adam(m_w_hat, v_w_hat, epsilon), modify_deltas_adam(m_b_hat, v_b_hat, epsilon),l2_regularization_constant)

    return Weights, Biases


def modify_deltas_nadam(m_theta_hat, v_theta_hat, delta_theta, beta1, epsilon, itr):
    factor = (1-beta1)/(1-np.power(beta1,itr+1))
    w_t_temp = accumulate_history(m_theta_hat, delta_theta, prev_factor=beta1, current_factor=factor)
    temp = []
    for  i in range(len(m_theta_hat)):
        temp.append(w_t_temp[i]/(np.sqrt(v_theta_hat[i])+epsilon))

    return temp


def gradient_descent_nadam(X, Y, learning_rate, number_of_layers,  batch_size, Weights, Biases, activation_function, l2_regularization_constant, beta1, beta2, epsilon, momentum=0, beta=0, loss_type="cross_entropy"):

    itr = 0
    v_t_weights = [np.zeros_like(weight) for weight in Weights]
    v_t_biases = [np.zeros_like(bias) for bias in Biases]
    m_weights = [np.zeros_like(weight) for weight in Weights]
    m_biases = [np.zeros_like(bias) for bias in Biases]
    m_w_hat = [np.zeros_like(weight) for weight in Weights]
    m_b_hat = [np.zeros_like(bias) for bias in Biases]
    v_w_hat = [np.zeros_like(weight) for weight in Weights]
    v_b_hat = [np.zeros_like(bias) for bias in Biases]

    for itr in tqdm(range(X.shape[0]//batch_size)):

        H, A, y_pred = forward_propagation(X[itr*batch_size:(itr+1)*batch_size].transpose(), Weights, Biases, number_of_layers, activation_function,batch_size)
        delta_Weights, delta_Biases = backward_propagation(H, A, Weights, Y[itr*batch_size:(itr+1)*batch_size].transpose(), y_pred, number_of_layers, activation_function, loss_type)

        m_weights = accumulate_history(m_weights,delta_Weights,prev_factor=beta1, current_factor=1-beta1)
        m_biases = accumulate_history(m_biases,delta_Biases, prev_factor=beta1, current_factor=1-beta1)

        v_t_weights = accumulate_history(v_t_weights,square_each_term(delta_Weights),prev_factor=beta2, current_factor=1-beta2)
        v_t_biases = accumulate_history(v_t_biases,square_each_term(delta_Biases), prev_factor=beta2, current_factor=1-beta2)

        m_w_hat = update_theta_hat_adam(m_weights,beta1,itr)
        m_b_hat = update_theta_hat_adam(m_biases,beta1,itr)
        
        v_w_hat = update_theta_hat_adam(v_t_weights,beta2,itr)
        v_b_hat = update_theta_hat_adam(v_t_biases,beta2,itr)

        Weights, Biases = update_weights_and_biases(learning_rate, Weights, Biases, modify_deltas_nadam(m_w_hat, v_w_hat, delta_Weights, beta1, epsilon, itr), modify_deltas_nadam(m_b_hat, v_b_hat, delta_Biases, beta1, epsilon, itr),l2_regularization_constant)

    return Weights, Biases
