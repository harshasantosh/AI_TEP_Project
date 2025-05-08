import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv1D, Flatten, Dropout
from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from ConfusionMatrix import confusion_matrix_multi, histogram
import gc
import os

#from Vote import all_predictions

# original dataset
unaug_dat = read_rdata('.\\TEP_FaultFree_Training.RData',
                       '.\\TEP_Faulty_Training.RData')

# no 3 9 or 15
# also no datapoints before sample = 20 because fault not introduced yet
unaug_dat = remove_bad_faults(unaug_dat)

# min max scale as in original paper
unaug_dat = data_scaler(unaug_dat)

# create multiple channels for data before and after the current datapoint which is in the middle


#window = 10
#step = 20

# as in original paper, split 60% training 40% testing
train_data, test_data = split_dataset(unaug_dat)
del unaug_dat
del train_data
gc.collect()
# convert to a form that CNNs can work on it


# One hot encode label


window = 15
step = 1
#train_X, train_Y = make_cnn_data(train_data, window=window, step=step)
test_X, test_Y = make_cnn_data(test_data, window=window, step=step)

enc = OneHotEncoder()
#enc.fit(train_Y.reshape(-1, 1))
#train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
enc.fit(test_Y.reshape(-1, 1))
test_labels_enc = enc.transform(test_Y.reshape(-1, 1))

# Assume predictions from the three models are stored in `predictions_dnn`, `predictions_cnn`, and `predictions_lstm`.
# These predictions are in one-hot encoded format.

def majority_voting(predictions):
    # Convert each prediction to class index (argmax)
    predicted_classes = np.argmax(predictions, axis=1)

    # Vote based on majority class (mode of the class indices)
    voted_class = np.argmax(np.bincount(predicted_classes))
    return voted_class

'''
model1_path = '.\\CNN_Multi_Hyper\\cp_15_1_tanh_kern_3_opt_adam_drop_0.ckpt.keras'
model1 = tf.keras.models.load_model(model1_path)
model2_path = '.\\CNN_Multi_Hyper\\cp_15_1_selu_kern_3_opt_adam_drop_0.ckpt.keras'
model2 = tf.keras.models.load_model(model2_path)
model3_path = '.\\CNN_Multi_Hyper\\cp_15_1_tanh_kern_5_opt_adam_drop_0.ckpt.keras'
model3 = tf.keras.models.load_model(model3_path)

predictions_1 = model1.predict(test_X)
predictions_2 = model2.predict(test_X)
predictions_3 = model3.predict(test_X)
'''

prediction_list = []
#model_list = []
path = '.\\Resnet_and_15\\'
for file in os.listdir(path):
    print(file)
    try:
        model = tf.keras.models.load_model(path+file)
        prediction = model.predict(test_X)

        #model_list.append(model)
        prediction_list.append(prediction)
    except Exception as ex:
        print(ex)



# Combine the predictions from all models
#all_predictions = np.stack([predictions_1, predictions_2, predictions_3], axis=1)
all_predictions = np.stack([x for x in prediction_list], axis=1)


#####
# Convert one-hot predictions to class indices (argmax)
predicted_classes = np.argmax(all_predictions, axis=2)  # shape (num_samples, 3)

# Class indices used by the model (0 to 17, since classes 3, 9, and 15 are missing)
valid_classes = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]

# Create a dictionary that maps the model's class index (0 to 17) to the actual class labels
model_to_actual_class_mapping = {i: valid_classes[i] for i in range(len(valid_classes))}

# Perform majority voting by using np.bincount for each sample
final_predictions = []
for i in range(predicted_classes.shape[0]):  # Loop over each sample
    # Count votes per class (from the predicted classes)
    class_votes = np.bincount(predicted_classes[i], minlength=len(valid_classes))

    # Find the class with the most votes (model index)
    voted_model_class_index = np.argmax(class_votes)

    # Map the model's predicted class index to the actual class
    final_class = model_to_actual_class_mapping[voted_model_class_index]
    final_predictions.append(final_class)

# Convert final predictions to a numpy array
final_predictions = np.array(final_predictions)

####

# Perform majority voting
#final_predictions = [majority_voting(all_predictions[:, i]) for i in range(all_predictions.shape[1])]

print(final_predictions)
print(len(final_predictions))

#prediction = enc.inverse_transform(final_predictions)
true = enc.inverse_transform(test_labels_enc)
print(type(true))
print(true.size)

print("Accuracy:", accuracy_score(true, final_predictions))

# Confusion matrix to show accuracies
#confusion_matrix_multi(true, prediction, 'Confusion Matrix')
histogram(true, final_predictions, 'Class Accuracy Histogram')


bin_true = []
bin_pred = []

for t in true:
    if t > 0:
        bin_true.append(1)
    else:
        bin_true.append(0)

for p in final_predictions:
    if p > 0:
        bin_pred.append(1)
    else:
        bin_pred.append(0)

#print(bin_true)
#print(bin_pred)

print("Accuracy:", accuracy_score(bin_true, bin_pred))