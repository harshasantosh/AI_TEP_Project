# Import the required libraries
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def confusion_matrix_binary(true, prediction, title):
    _, ax = plt.subplots()
    cm = confusion_matrix(true, prediction)
    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_ylim(len(cm), 0)
    plt.tight_layout()
    plt.show()


def confusion_matrix_multi(true, prediction, title):
    _, ax = plt.subplots()
    '''
    new_true = []
    for x in true:
        value = x[0]
        if value > 3:
            value += 1
        if value > 9:
            value += 1
        if value > 15:
            value += 1


        new_true.append([value])
    true = np.asarray(new_true)

    new_prediction = []
    for x in prediction:
        value = x[0]
        if value > 3:
            value += 1
        if value > 9:
            value += 1
        if value > 15:
            value += 1


        new_prediction.append([value])
    prediction = np.asarray(new_prediction)
    '''

    cm = confusion_matrix(true, prediction, normalize='true')
    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_ylim(len(cm), 0)
    plt.tight_layout()
    plt.show()


def histogram(true, prediction, title):
    _, ax = plt.subplots()
    '''
    new_true = []
    for x in true:
        value = x[0]
        if value > 3:
            value += 1
        if value > 9:
            value += 1
        if value > 15:
            value += 1


        new_true.append([value])
    true = np.asarray(new_true)

    new_prediction = []
    for x in prediction:
        value = x[0]
        if value > 3:
            value += 1
        if value > 9:
            value += 1
        if value > 15:
            value += 1


        new_prediction.append([value])
    prediction = np.asarray(new_prediction)
    '''
    class_accuracies = []
    num_classes = 21
    # Loop over each class and calculate accuracy
    for class_index in range(num_classes):
        # Get the indices where the true label is this class
        true_class_indices = np.where(true == class_index)[0]

        # Get the predicted labels for these indices
        predicted_class_labels = prediction[true_class_indices]

        # Calculate accuracy for this class
        class_accuracy = np.sum(predicted_class_labels == class_index) / len(true_class_indices)

        class_accuracies.append(class_accuracy)
        print(f"Accuracy for class {class_index}: {class_accuracy:.4f}")
    #plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], ['0', '1', '2', '3', '4', '5', '6',
    #                                                                            '7', '8', '9', '10', '11', '12', '13', '14', '15',
    #                                                                            '16', '17',
    #                                                                            '18', '19', '20'])
    # Optionally, you can plot the per-class accuracies
    #plt.bar(range(num_classes), class_accuracies)
    #plt.xlabel('Class')
    #plt.ylabel('Accuracy')
    #plt.title('Per-Class Accuracy')
    #plt.show()