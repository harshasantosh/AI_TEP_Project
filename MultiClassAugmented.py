# This is an implementation of Fault detection and classification using artificial neural networks's
# multiclass neural network with the shape described as 102-102-50-40-18
# the only change made is in the batch size. The original batch size was sounded like it was
# training set / 50, but this was ruinous to our ability to get the results they reported.
# It's unclear what exactly they meant.
#
# Our neural network gets very similar results to the paper's, sometimes better even.

from keras.layers import Input, Dense
import tensorflow as tf
from ConfusionMatrix import confusion_matrix_multi
from sklearn.metrics import accuracy_score

from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder

# original dataset
unaug_dat = read_rdata('C:\\Users\\Arthur\\PycharmProjects\\AI_Poject\\TEP_FaultFree_Training.RData',
           'C:\\Users\\Arthur\\PycharmProjects\\AI_Poject\\TEP_Faulty_Training.RData')

unaug_dat = remove_bad_faults(unaug_dat)

# augmented. in the paper, the augmented multiclass dataset has a window of size 2
aug_dat = combine_dataset(unaug_dat, 2)

# train and test dataset
train_data, test_data = split_dataset(aug_dat)

# data before 3 is either useless or the class label
train_X = train_data.iloc[:,3:].values
test_X = test_data.iloc[:,3:].values

# labels
train_Y = train_data['faultNumber'].values
test_Y = test_data['faultNumber'].values

# One hot encode
enc = OneHotEncoder()
enc.fit(train_Y.reshape(-1, 1))
train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
enc.fit(test_Y.reshape(-1, 1))
test_labels_enc = enc.transform(test_Y.reshape(-1, 1))

# Input is of size 52
checkpoint_path = "MulticlassAugmented\\cp_rearranged_preprocess.ckpt.keras"
load = False
if load:
    model = tf.keras.models.load_model(checkpoint_path)
else:
    # the network is described as 102-102-50-40-18
    # this description makes no sense, either it's 52-102-102-50-40-18 or 102-102-50-40 with the 52 and 18
    # input / output implied
    # this is what we implemented
    # also possible is something like 52 102 102 50 40 18 18
    model = tf.keras.Sequential([
        Input(shape=(train_X.shape[1],)),
        Dense(102, activation='relu'),
        Dense(102, activation='relu'),
        Dense(50, activation='relu'),
        Dense(40, activation='relu'),
        Dense(train_labels_enc.shape[1], activation='softmax')
    ])

    # original paper specified adam. It didn't specify crossentropy, but it didn't specify otherwise
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=False,
                                                     verbose=1)


    batch_size = 256
    #fit model
    history = model.fit(train_X, train_labels_enc.todense(), epochs=400, batch_size=batch_size,
                        validation_data=(test_X, test_labels_enc.todense()),
                        callbacks=[cp_callback])


prediction = enc.inverse_transform(model.predict(test_X, verbose=0))
true = enc.inverse_transform(test_labels_enc)

print("Accuracy:", accuracy_score(true, prediction))

# Confusion matrix to show accuracies
confusion_matrix_multi(true, prediction, 'Confusion Matrix')

