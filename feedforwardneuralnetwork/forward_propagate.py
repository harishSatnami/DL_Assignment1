# forward propagation 

from pre_activation_functions import pre_activation
from activation_functions import sigmoid_vector, relu_vector, tanh_vector, softmax

activation = sigmoid_vector

def forward_propagation(input, Weights, Biases, number_of_layers):
    A = []
    H = [input]
    for i in range(number_of_layers-1):
        A.append(pre_activation(Weights[i],H[i],Biases[i]))
        H.append(activation(A[i]))

    A.append(pre_activation(Weights[-1], H[-1], Biases[-1]))

    y_pred = softmax(A[-1])

    return H, A, y_pred