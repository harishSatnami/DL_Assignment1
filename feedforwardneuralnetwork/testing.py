from forward_propagate import forward_propagation

def validate(X, Weights, Biases):
    # some calculations
    H, A, Y_pred = forward_propagation(X, Weights=Weights, Biases=Biases, number_of_layers=len(Weights))
    return Y_pred


