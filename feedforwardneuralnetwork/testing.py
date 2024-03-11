from forward_propagate import forward_propagation

def validate(X, Weights, Biases, activation_function):
    # some calculations
    H, A, Y_pred = forward_propagation(X, Weights=Weights, Biases=Biases, number_of_layers=len(Weights)+1, activation_function=activation_function)
    return Y_pred


