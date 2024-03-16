import wandb
import numpy as np
from keras.datasets import fashion_mnist

from utility import get_accuracy, cross_entropy_loss, mean_squared_error, random_initialize, xavier_initialize
from gradient_descent_FFNN import gradient_descent_adam, gradient_descent_mini_batch, gradient_descent_momentum_based, gradient_descent_nadam, gradient_descent_nesterov_accelarated, gradient_descent_RMSProp
from testing import validate


# sweep configuration

sweep_config = {
    'method': 'bayes',
    'name' : 'DL_ASSIGNMENT_1',
    'metric': {
      'name': 'validation_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.1, 0.01, 0.001, 0.0001, 0.00001]
        },
        'activation_function': {
            'values': ['sigmoid', 'relu', 'tanh']
        },
        'batch_size': {
            'values': [25, 50, 100, 150, 200]
        },
        'epochs': {
            'values': [5, 10]
        },
        'num_of_hidden_layer': {
            'values': [3, 4, 5]
        },
        'size_of_layer': {
            'values': [32, 64, 128]
        },
        'optimizer': {
            'values': ['sgd', 'mbgd', 'nagd', 'rmsprop', 'adam', 'nadam']
        },
        'weight_init_type': {
            'values': ['random', 'xavier']
        },
        'l2_reg_constant': {
            'values' : [0, 0.0005, 0.5]
        },
        'loss_type': {
            'values' : ["mse"]
        }
    }
}

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
train_size = trainX.shape[0] *0.9
validateX = trainX[train_size:]
validateY = trainY[train_size:]
trainX = trainX[:train_size]
trainY = trainY[:train_size]

trainx = trainX.reshape(trainX.shape[0],-1)/255
validatex = validateX.reshape(validateX.shape[0],-1)/255
testx = testX.reshape(testX.shape[0],-1)/255

# output dataset conversion One hot encoding
trainy = [np.zeros(10) for i in range(trainX.shape[0])]
validatey = [np.zeros(10) for i in range(validateX.shape[0])]
testy = [np.zeros(10) for i in range(testX.shape[0])]

for i in range(trainX.shape[0]):
    trainy[i][trainY[i]] = 1

for i in range(validateX.shape[0]):
    validatey[i][validateY[i]] = 1

for i in range(testX.shape[0]):
    testy[i][testY[i]] = 1

trainy = np.array(trainy)
testy = np.array(testy)

X = trainx
Y = trainy


def train_model_with_wandb():
    
    config=wandb.config

    epochs = config.epochs
    num_of_hidden_layers = config.num_of_hidden_layer
    size_of_layers = config.size_of_layer
    learning_rate = config.learning_rate
    optimizer = config.optimizer
    batch_size = config.batch_size
    l2_regularization_constant = config.l2_reg_constant
    weight_init_type = config.weight_init_type
    activation_function = config.activation_function
    beta = 0.9
    beta1 = 0.999
    epsilon = 0.0001
    loss_type = config.loss_type
        
    
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
    print("beta",beta)
    print("beta1:", beta1)
    print("epsilon:", epsilon)
    print("loss type:",loss_type)

    if weight_init_type=="random":
        initialize = random_initialize
    else:
        initialize = xavier_initialize

    if optimizer=="mini_batch" or optimizer=="sgd":
        gradient = gradient_descent_mini_batch
    elif optimizer=="mbgd":
        gradient = gradient_descent_momentum_based
    elif optimizer=="rmsprop":
        gradient = gradient_descent_RMSProp
    elif optimizer=="nagd":
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
                                   momentum=beta,
                                   beta=beta,
                                   beta1=beta, 
                                   beta2=beta1,
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



def run_wandb_sweep(project, entity, sweep_count=10):
    wandb.login()
    wandb.init(project=project, entity=entity)
    sweep_id = wandb.sweep(sweep=sweep_config, project='DL_Assignment_1')
    wandb.agent(sweep_id, function=train_model_with_wandb, count = sweep_count)
    wandb.finish()