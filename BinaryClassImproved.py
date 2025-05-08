# This is an implementation of Fault detection and classification using artificial neural networks's
# fault detect / binary classifier neural network with the shape described as 52-52-25-2-2
# the only change made is in the batch size. The original batch size was sounded like it was
# training set / 50, but this was ruinous to our ability to get the results they reported.
# It's unclear what exactly they meant.
#
# Our neural network gets very similar results to the paper's, sometimes better even.
import numpy as np
from keras.layers import Input, Dense
import tensorflow as tf
from ConfusionMatrix import confusion_matrix_binary
from sklearn.metrics import accuracy_score, confusion_matrix
from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import keras

# original dataset
unaug_dat = read_rdata('C:\\Users\\Arthur\\PycharmProjects\\AI_Poject\\TEP_FaultFree_Training.RData',
           'C:\\Users\\Arthur\\PycharmProjects\\AI_Poject\\TEP_Faulty_Training.RData')

unaug_dat = remove_bad_faults(unaug_dat)

#pos = unaug_dat[unaug_dat['faultNumber']!=0].shape[0]
#neg = unaug_dat[unaug_dat['faultNumber']==0].shape[0]
#total = pos + neg

# augmented. in the paper, the augmented binary dataset has a window of size 3
aug_dat = combine_dataset(unaug_dat, 3)

# train and test dataset
train_data, test_data = split_dataset(aug_dat)

# balancing the dataset was not stated to have been done in the original paper, but
# the model got too much out of guessing 1 every time without this.
#train_data = balance_data(train_data)

# data before 3 is either useless or the class label
train_X = train_data.iloc[:,3:].values
test_X = test_data.iloc[:,3:].values

# convert to binary labels
train_data.loc[train_data['faultNumber'] > 1, 'faultNumber'] = 1
test_data.loc[test_data['faultNumber'] > 1, 'faultNumber'] = 1

# labels
train_Y = train_data['faultNumber'].values
test_Y = test_data['faultNumber'].values



# One hot encode
#enc = OneHotEncoder()
#enc.fit(train_Y.reshape(-1, 1))
#train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
#enc.fit(test_Y.reshape(-1, 1))
#test_labels_enc = enc.transform(test_Y.reshape(-1, 1))

checkpoint_path = "BinaryClassReluTruncatedGeluWeighted\\cp.ckpt.keras"
load=False
if load:
    model = tf.keras.models.load_model(checkpoint_path)
else:
    # Input is of size 52
    # the network is described as 52-52-25-2-2
    # In light of the '102-102-50-40-18' the exact dimesnions are difficult to know
    # We went with 52-52-52-25-2-2, but also possible are 52-52-25-2-2 and 52-52-52-25-2-2-2
    # as strange as it is, they seem to use softmax and cross entropy
    model = tf.keras.Sequential([
        Input(shape=(train_X.shape[1],)),
        Dense(52, activation='gelu'),
        Dense(52, activation='gelu'),
        Dense(25, activation='gelu'),
        #Dense(2, activation='relu'),
        #Dense(2, activation='softmax')
        Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam()#(learning_rate=0.00001)

    # original paper specified adam. It didn't specify crossentropy, but it didn't specify otherwise
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=False,
                                                     verbose=1)

    batch_size = 256
    # there are far fewer 0s than 1s, this helps the model not just guess 1 every time

    # 0s outnumbered 17 to 1
    class_weight = {0: 17, 1: 1}
    # fit model
    history = model.fit(train_X, train_Y,  # train_labels_enc.todense(),
                        epochs=400, batch_size=batch_size,
                        validation_data=(test_X, test_Y),  # test_labels_enc.todense()),
                        callbacks=[cp_callback], class_weight=class_weight)


#a = model.evaluate(test_X, test_labels_enc.todense())
#print(a)

pred_Y = model.predict(test_X)
pred_Y = tf.round(pred_Y)  # Convert predictions to binary (0 or 1)

cm = confusion_matrix(test_Y, pred_Y)

class_0_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1])
class_1_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print("Class 0 Accuracy:", class_0_accuracy)
print("Class 1 Accuracy:", class_1_accuracy)