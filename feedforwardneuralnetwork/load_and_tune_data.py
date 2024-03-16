from keras.datasets import fashion_mnist, mnist
import numpy as np


def load_and_tune_dataset(dataset):
    if dataset=="mnist":
        data_to_load = mnist
    else:
        data_to_load = fashion_mnist

    (trainX, trainY), (testX, testY) = data_to_load.load_data()
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
    
    return trainx, trainy, validatex, validatey, testx, testy
