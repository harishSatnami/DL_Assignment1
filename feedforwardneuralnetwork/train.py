
# imports
import numpy as np
from keras.datasets import fashion_mnist, mnist
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import argparse

from load_and_tune_data import load_and_tune_dataset
from training import train_model
from testing import validate
from plots import plot_confusion_matrix
from utility import get_accuracy, cross_entropy_loss

from wandb_sweep import run_wandb_sweep

def run_model():
    
    parser = argparse.ArgumentParser(description="Command line arguments for FEED FORWARD NEURAL NETWORK")
    
    parser.add_argument("-wp","--wandb_project",type=str,default="Dl_Assignment_1", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we","--wandb_entity",type=str,default="cs23m025", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d","--dataset",type=str,default="fashion_mnist", choices= ['mnist', 'fashion_mnist'])
    parser.add_argument("-e","--epochs",type=int,default=10, help="Number of epochs to train neural network.")
    parser.add_argument("-b","--batch_size",default=150,type=int, help="Batch size used to train neural network.")
    parser.add_argument("-l","--loss",default="cross_entropy",type=str, choices = ['mean_squared_error', 'cross_entropy'])
    parser.add_argument("-o","--optimizer",default="nadam",type=str, choices= ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument("-lr","--learning_rate",default=0.001,type=float, help="Learning rate used to optimize model parameters")
    parser.add_argument("-m","--momentum",default=0.9,type=float, help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta","--beta",default=0.9,type=float, help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1","--beta1",default=0.9,type=float, help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2","--beta2",default=0.999, type=float,help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps","--epsilon",default=0.00001,type=float, help="Epsilon used by optimizers.")
    parser.add_argument("-w_d","--weight_decay",default=0.0005,type=float, help="Weight decay used by optimizers.")
    parser.add_argument("-w_i","--weight_init",default="xavier",type=str, choices= ['random', 'Xavier'])
    parser.add_argument("-nhl","--num_layers",default=3,type=int, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz","--hidden_size",default=128, type=int,help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a","--activation",default="tanh",type=str, choices= ['identity', 'sigmoid', 'tanh', 'ReLU'])
    parser.add_argument("-rs","--run_sweep",default="False",type=str, choices= ['True', 'False', 'Yes', 'No'], help="Parameter to check whether to run sweep or not")
    parser.add_argument("-swc","--sweep_count",default=20,type=int, help="Number of runs for the sweep")
    
    
    args = parser.parse_args()
    
    project = args.wandb_project
    entity = args.wandb_entity
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size 
    loss_type = args.loss.lower()
    optimizer = args.optimizer.lower()
    learning_rate = args.learning_rate
    momentum = args.momentum
    beta = args.beta
    beta1 = args.beta1
    beta2 = args.beta2
    epsilon = args.epsilon
    l2_regularization_constant = args.weight_decay
    weight_init_type = args.weight_init.lower()
    num_of_hidden_layers = args.num_layers
    size_of_layers = args.hidden_size
    activation_function = args.activation.lower()
    
    run_sweep = args.run_sweep.lower()
    sweep_count = args.sweep_count
    
    if run_sweep=="true" or run_sweep=="yes": 
        # not fully tested 
        pass
        # run_wandb_sweep(project=project, entity=entity, sweep_count=sweep_count)
        # return
    
    wandb.init(project=project, entity=entity)
    
    
    # Taking input data and normalizing data

    trainx, trainy, validatex, validatey, testx, testy = load_and_tune_dataset(dataset)

    X = trainx
    Y = trainy
    
    #--------------------------------------------------------------------------------------

    Weights, Biases = train_model(
                        X=X,
                        Y=Y,
                        validatex=validatex,
                        validatey=validatey,
                        epochs=epochs,
                        num_of_hidden_layers=num_of_hidden_layers,
                        size_of_layers=size_of_layers,
                        learning_rate=learning_rate,
                        optimizer=optimizer,
                        batch_size=batch_size,
                        l2_regularization_constant=l2_regularization_constant,
                        weight_init_type=weight_init_type,
                        activation_function=activation_function,
                        beta=beta,
                        beta1=beta1,
                        beta2=beta2,
                        momentum=momentum,
                        epsilon=epsilon,
                        loss_type=loss_type
                    )

    Y_predict_test = validate(testx, Weights, Biases, activation_function)
    test_accuracy_percentage = get_accuracy(testy, Y_predict_test)
    test_loss = cross_entropy_loss(testy, Y_predict_test)
    print("Test accuracy : ", test_accuracy_percentage)
    print("Test loss : ", test_loss)
    label_display_name = None
    if dataset=="mnist":
        label_display_name = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    elif dataset=="fashion_mnist":
        label_display_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    plot_confusion_matrix(testy, Y_predict_test, label_display_name)


wandb.login()
if __name__=="__main__":
    run_model()
wandb.finish()
