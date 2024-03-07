# forward propagation 

from pre_activation_functions import pre_activation
from activation_functions import sigmoid_vector, relu_vector, tanh_vector, softmax

activation = sigmoid_vector

def forward_propagation(input, Weights_w, Weights_l, Biases_b, Biases_l, number_of_layers):
    A = []
    H = [input]
    for i in range(1,number_of_layers):
        A.append(pre_activation(Weights_w[i],H[i-1],Biases_b[i]))
        H.append(activation(A[i-1]))

    A.append(pre_activation(Weights_l, H[-1], Biases_l))

    y_pred = softmax(A[-1])

    return H, A, y_pred