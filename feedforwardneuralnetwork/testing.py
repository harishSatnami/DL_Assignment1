from forward_propagate import forward_propagation

def validate(X, Weights, Biases, activation_function):
    H, A, Y_pred = forward_propagation(X.transpose(), Weights=Weights, Biases=Biases, number_of_layers=len(Weights)+1, activation_function=activation_function, batch_size=X.shape[0])
    return Y_pred.transpose()


