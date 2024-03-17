# DL_Assignment1
Assignment 1 on Back-propagation Deep Learning

## Introduction
This is Assignment 1 for the course Fundamental of Deep Learning (IIT MADRAS), where we do model training for fashion_mnist dataset by extensively doing experiments with different hyper parameters. 
The Assignment covers the understanding of Feed Forward Neural Network using forward and backward propogation. In the Assignment we have implemented various gradient descent algorithms and try to find out the best algorithm-parameters combination for which we get the maximum accuracy.

## Requirements
This project requires following 

- Python
- WandB
- Numpy
- Tensorflow (for dataset)
- tqdm


## Understanding the datasets used
In this project following datasets are used (we are importing it from keras, Below links are just to check about the datasets)

- fashion_mnist https://www.tensorflow.org/datasets/catalog/fashion_mnist
    Total number of datapoints = 70000
    Total number of training datapoints = 60000
    Total number of testing datapoints = 10000
    Shape of each datapoint = 28 * 28
    Number of output classes = 10
    Labels of output classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


- mnist https://www.tensorflow.org/datasets/catalog/mnist
    Total number of datapoints = 70000
    Total number of training datapoints = 60000
    Total number of testing datapoints = 10000
    Shape of each datapoint = 28 * 28
    Number of output classes = 10
    Labels of output classes = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


## Understanding packages used
- tqdm - Used for showing progress bar
- numpy - Used for numeric calculations( specially for array and matrics)
- keras - Used for downloading the datasets
- matplotlib - Used for plots
- wandb - Used for visualization of different parameters

## Installation

1. Clone the repository:

   bash
   git clone https://github.com/yourusername/fashion-mnist-neural-network.git
   

2. Install dependencies:

   bash
   pip install -r requirements.txt
   


### Different optimizers supported by this project are :
* sgd
* momentum
* nag
* rmsprop
* adam
* nadam


## EXPLAINATION OF PY FILES

### forward_propagate.py 
def forward_propagation(X, Weights, Biases, number_of_layers, activation_function, batch_size):<br>
    """
    Perform forward propagation through a neural network.

    Parameters:
    - X : numpy array
        Input data of shape (input_size, batch_size).
    - Weights : list of numpy arrays
        List of weight matrices for each layer of the neural network.
    - Biases : list of numpy arrays
        List of bias vectors for each layer of the neural network.
    - number_of_layers : int
        Number of layers in the neural network, excluding the input layer.
    - activation_function : str
        Activation function to be used. Choices are ["relu", "tanh", "identity", "sigmoid"].
    - batch_size : int
        Batch size for processing the input data.

    Returns:
    - H : list of numpy arrays
        List of hidden layer activations.
    - A : list of numpy arrays
        List of pre-activation values for each layer.
    - y_pred : numpy array
        Predicted probabilities for each class for each input in the batch, of shape (batch_size, num_classes).
    """


### backward_propagate.py
def backward_propagation(H, A, W, y_actual, y_pred, number_of_layers, activation_function, loss_type):<br>
    """
    Perform backward propagation to compute gradients of the loss with respect to weights and biases.

    Parameters:
    - H : list of numpy arrays
        List of hidden layer activations.
    - A : list of numpy arrays
        List of pre-activation values for each layer.
    - W : list of numpy arrays
        List of weight matrices for each layer of the neural network.
    - y_actual : numpy array
        Actual labels for the input data, of shape (batch_size, num_classes).
    - y_pred : numpy array
        Predicted probabilities for each class for each input in the batch, of shape (batch_size, num_classes).
    - number_of_layers : int
        Number of layers in the neural network, excluding the input layer.
    - activation_function : str
        Activation function used in the forward propagation. Choices are ["relu", "tanh", "identity", "sigmoid"].
    - loss_type : str
        Type of loss function used. Choices are ["mse", "mean_squared_error", "cross_entropy"].

    Returns:
    - delta_W : list of numpy arrays
        List of gradients of the loss with respect to weights for each layer.
    - delta_B : list of numpy arrays
        List of gradients of the loss with respect to biases for each layer.
    """

## Usage

 Train the neural network:

   Run the following command to train the neural network:

   bash
   python train.py --wandb_project myprojectname --wandb_entity myname --dataset fashion_mnist --epochs 10 --batch_size 50 --loss cross_entropy --optimizer nadam --learning_rate 0.0001 --momentum 0.9 --beta 0.9 --beta1 0.9 --beta2 0.999 --epsilon 0.000001 --weight_decay 0 --weight_init Xavier --num_layers 5 --hidden_size 128 --activation ReLU

   
   You can pass the hyperparameters in command line before training.

## INSTRUCTIONS ON HOW TO RUN 

* Create a wandb Account before running train.py file.
* Give the api key to your account when prompted.
* install packages as mentioned in the Installation section
  
The following table contains the arguments supported by the train.py file
|Name|	Default Value|	Description|
|:----:| :---: |:---:|
|-wp, --wandb_project	|myprojectname	|Project name used to track experiments in Weights & Biases dashboard|
|-we, --wandb_entity|	myname	|Wandb Entity used to track experiments in the Weights & Biases dashboard.|
|-d, --dataset|	fashion_mnist	|choices: ["mnist", "fashion_mnist"]|
|-e, --epochs	|10	|Number of epochs to train neural network.|
|-b, --batch_size|	50	|Batch size used to train neural network.|
|-l, --loss	|cross_entropy	|choices: ["mean_squared_error", "cross_entropy"]|
|-o, --optimizer|	nadam	|choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]|
|-lr, --learning_rate	|0.0001	|Learning rate used to optimize model parameters|
|-m, --momentum|	0.9	|Momentum used by momentum and nag optimizers.|
|-beta, --beta|	0.9|	Beta used by rmsprop optimizer|
|-beta1, --beta1|	0.9|	Beta1 used by adam and nadam optimizers.|
|-beta2, --beta2|	0.99	|Beta2 used by adam and nadam optimizers.|
|-eps, --epsilon|	0.000001	|Epsilon used by optimizers.|
|-w_d, --weight_decay|	0	|Weight decay used by optimizers.|
|-w_i, --weight_init	|Xavier	|choices: ["random", "Xavier"]|
|-nhl, --num_layers|	5	|Number of hidden layers used in feedforward neural network.|
|-sz, --hidden_size|	128	|Number of hidden neurons in a feedforward layer.|
|-a, --activation	|ReLU	|choices: ["identity", "sigmoid", "tanh", "ReLU"]|