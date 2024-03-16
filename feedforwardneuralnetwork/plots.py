import numpy as np
import matplotlib.pyplot as plt
import wandb


# Display few images
def plot_images(trainX,trainY):
    cnt = 0
    label_display_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    label = []

    for i in range(1, trainX.shape[0]):

        if trainY[i] not in label:
            cnt = cnt + 1
            label.append(trainY[i])
            plt.subplot(2, 5, cnt)
            # Insert ith image with the color map 'grap'
            plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
            plt.title(label_display_name[trainY[i]])

        if cnt == 10:
            break

    # Display the entire plot
    plt.show()


def plot_confusion_matrix( y_actual,y_pred, label_display_name):
    
    if not label_display_name:
        label_display_name = [str(i) for i in range(len(y_actual[0]))]
    y_pred = np.argmax(y_pred, axis=1)
    y_actual = np.argmax(y_actual, axis=1)

    # label_display_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                           y_true=y_actual,
                                                           preds=y_pred,
                                                           class_names=label_display_name,
                                                           title="Confusion Matrix for Test Data"
                                                           )})