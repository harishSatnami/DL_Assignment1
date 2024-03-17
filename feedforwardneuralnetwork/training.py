import numpy as np
import wandb

from utility import random_initialize, xavier_initialize, mean_squared_error, cross_entropy_loss, get_accuracy
from testing import validate
from gradient_descent_FFNN import gradient_descent_mini_batch, gradient_descent_momentum_based, gradient_descent_RMSProp, gradient_descent_nesterov_accelarated, gradient_descent_adam, gradient_descent_nadam


def train_model(X, Y,validatex,validatey, epochs=1, num_of_hidden_layers=1, size_of_layers=4, learning_rate=0.1, optimizer="sgd", batch_size=4, l2_regularization_constant=0.001, weight_init_type="random", activation_function="sigmoid", beta = 0,beta1=0, epsilon=1e-10, beta2=0, loss_type="cross_entropy", momentum=0.9, **params):
    run_name = "{}_lr{}_bs{}_hl{}_hlsize{}_{}_{}_epochs{}_{}".format(optimizer, learning_rate, batch_size, num_of_hidden_layers, size_of_layers, activation_function,weight_init_type, epochs,loss_type)
    wandb.run.name=run_name

    print("number of training datapoints:",X.shape[0])
    print("number of epochs:", epochs)
    print("number of hidden layers:", num_of_hidden_layers)
    print("size of hidden layers:", size_of_layers)
    print("learning rate:", learning_rate)
    print("optimizer:", optimizer)
    print("batch_size:", batch_size)
    print("l2 regularization constant:", l2_regularization_constant)
    print("weights and biases initialization type:", weight_init_type)
    print("activation function:", activation_function)
    print("momentum:",momentum)
    print("beta",beta)
    print("beta1:", beta1)
    print("beta2:", beta2)
    print("epsilon:", epsilon)
    print("loss type:",loss_type)

    if weight_init_type=="random":
        initialize = random_initialize
    else:
        initialize = xavier_initialize

    if optimizer=="mini_batch" or optimizer=="sgd":
        gradient = gradient_descent_mini_batch
    elif optimizer=="mbgd" or optimizer=="momentum":
        gradient = gradient_descent_momentum_based
    elif optimizer=="rmsprop":
        gradient = gradient_descent_RMSProp
    elif optimizer=="nagd" or optimizer=="nag":
        gradient = gradient_descent_nesterov_accelarated
    elif optimizer=="adam":
        gradient = gradient_descent_adam
    elif optimizer=="nadam":
        gradient = gradient_descent_nadam
    else:
        gradient = gradient_descent_mini_batch

    Weights, Biases = initialize(num_of_hidden_layers+2,size_of_layers,Y.shape[1], X.shape[1])

    for epoch in range(epochs):
        print("Epoch number", epoch+1, " started")
        Weights, Biases = gradient(X=X, 
                                   Y=Y, 
                                   learning_rate=learning_rate,
                                   number_of_layers=num_of_hidden_layers+2,
                                   batch_size=batch_size, 
                                   Weights=Weights, 
                                   Biases=Biases, 
                                   activation_function=activation_function, 
                                   l2_regularization_constant=l2_regularization_constant,
                                   momentum=momentum,
                                   beta=beta,
                                   beta1=beta1, 
                                   beta2=beta2,
                                   epsilon=epsilon, 
                                   loss_type=loss_type)
        print("Epoch ",epoch+1, " finished.")

        Y_predict = validate(validatex,Weights,Biases,activation_function)
        Y_predict_train = validate(X, Weights, Biases,activation_function)

        if loss_type == "mse" or "mean_squared_error":
            validation_loss = mean_squared_error( validatey,Y_predict)
            training_loss = mean_squared_error( validatey,Y_predict)
        else:
            validation_loss = cross_entropy_loss( validatey,Y_predict)
            training_loss = cross_entropy_loss(Y,Y_predict_train)

        validation_accuracy = get_accuracy(validatey,Y_predict)
        training_accuracy = get_accuracy(Y,Y_predict_train)

        print("training accuracy after epoch ",epoch+1,":",training_accuracy,"%")
        print("training Loss (cross entropy)",training_loss)
        print("validation accuracy after epoch ",epoch+1,":",validation_accuracy,"%")
        print("Validation Loss (",loss_type,") : ",validation_loss)

        wandb.log({'training_loss': training_loss, 'validation_loss': validation_loss, 'training_accuracy': training_accuracy, 'validation_accuracy': validation_accuracy, 'epoch_number': epoch+1})

    return Weights, Biases