# This is an implementation of Fault detection and classification using artificial neural networks's
# fault detect / binary classifier neural network with the shape described as 52-52-25-2-2
# the only change made is in the batch size. The original batch size was sounded like it was
# training set / 50, but this was ruinous to our ability to get the results they reported.
# It's unclear what exactly they meant.
#
# Our neural network gets very similar results to the paper's, sometimes better even.
from threading import activeCount

import numpy as np
import pandas as pd
from keras.layers import Input, Dense
import tensorflow as tf

from BinaryClassImproved import pred_Y
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

# augmented. in the paper, the augmented binary dataset has a window of size 3
aug_dat = combine_dataset(unaug_dat, 3)

# train and test dataset
train_data, test_data = split_dataset(aug_dat)

# data before 3 is either useless or the class label
train_X = train_data.iloc[:,3:].values
test_X = test_data.iloc[:,3:].values

#enc = OneHotEncoder()
load=False
models = []
if load:
    offset = 0
    for vs_label in range(1,21):
        if vs_label == 3 or vs_label == 9 or vs_label == 15:
            print(vs_label, 'does not exist')
            continue

        checkpoint_path = f"BinarylassOneOnOne\\cp_binary_fault_{vs_label}.ckpt.keras"
        model = tf.keras.models.load_model(checkpoint_path)

        train_round = train_data.copy()
        test_round = test_data.copy()

        train_round = train_round[(train_round['faultNumber'] == 0) | (train_round['faultNumber'] == vs_label)]
        test_round = test_round[(test_round['faultNumber'] == 0) | (test_round['faultNumber'] == vs_label)]

        # convert to binary labels
        train_round.loc[train_round['faultNumber'] > 1, 'faultNumber'] = 1
        test_round.loc[test_round['faultNumber'] > 1, 'faultNumber'] = 1

        train_X = train_round.iloc[:, 3:].values
        test_X = test_round.iloc[:, 3:].values

        # labels
        train_Y = train_round['faultNumber'].values
        test_Y = test_round['faultNumber'].values

        # One hot encode
        #enc = OneHotEncoder()
        #enc.fit(train_Y.reshape(-1, 1))
        #train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
        #enc.fit(test_Y.reshape(-1, 1))
        #test_labels_enc = enc.transform(test_Y.reshape(-1, 1))

        #prediction = enc.inverse_transform(model.predict(test_X, verbose=0))
        pred_Y = model.predict(test_X)
        pred_y = tf.round(pred_Y)  # Convert predictions to binary (0 or 1)

        cm = confusion_matrix(test_Y, pred_Y)

        class_0_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        class_1_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        print("Class 0 Accuracy:", class_0_accuracy)
        print("Class 1 Accuracy:", class_1_accuracy)

        models.append(model)
else:
    for vs_label in range(21):
        checkpoint_path = f"BinarylassOneOnOne\\cp_binary_fault_{vs_label}.ckpt.keras"
        if vs_label == 3 or vs_label == 9 or vs_label == 15:
            print(vs_label, 'does not exist')
            continue
        train_round = train_data.copy()
        test_round = test_data.copy()

        train_round = train_round[(train_round['faultNumber'] == 0) | (train_round['faultNumber'] == vs_label)]
        test_round = test_round[(test_round['faultNumber'] == 0) | (test_round['faultNumber'] == vs_label)]


        # convert to binary labels
        train_round.loc[train_round['faultNumber'] > 1, 'faultNumber'] = 1
        test_round.loc[test_round['faultNumber'] > 1, 'faultNumber'] = 1

        print(train_round.head(5))
        print(train_round.tail(5))

        train_X = train_round.iloc[:, 3:].values
        test_X = test_round.iloc[:, 3:].values

        # labels
        train_Y = train_round['faultNumber'].values
        test_Y = test_round['faultNumber'].values

        # One hot encode
        #enc = OneHotEncoder()
        #enc.fit(train_Y.reshape(-1, 1))
        #train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
        #enc.fit(test_Y.reshape(-1, 1))
        #test_labels_enc = enc.transform(test_Y.reshape(-1, 1))


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
            Dense(2, activation='gelu'),
            #Dense(2, activation='softmax')
            Dense(1, activation='sigmoid')
        ])

        optimizer = keras.optimizers.Adam(learning_rate=0.00001)

        # original paper specified adam. It didn't specify crossentropy, but it didn't specify otherwise
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=False,
                                                         verbose=1)

        batch_size = 256
        # there are far fewer 0s than 1s, this helps the model not just guess 1 every time

        history = model.fit(train_X, train_Y#train_labels_enc.todense()
                            , epochs=400, batch_size=batch_size,
                        validation_data=(test_X, test_Y),#test_labels_enc.todense()),
                        callbacks=[cp_callback])

        pred_Y = model.predict(test_X)
        pred_y = tf.round(pred_Y)  # Convert predictions to binary (0 or 1)

        cm = confusion_matrix(test_Y, pred_Y)

        class_0_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        class_1_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        print("Class 0 Accuracy:", class_0_accuracy)
        print("Class 1 Accuracy:", class_1_accuracy)

        models.append(model)


combine_load = False
if combine_load:
    pass
else:
    predictions = pd.DataFrame()
    vs_label = 0
    for model in range(len(models)):
        vs_label+=1

        if vs_label == 3 or vs_label == 9 or vs_label == 15:
            vs_label += 1
            print(vs_label, 'does not exist')
            continue

        print(vs_label)

        checkpoint_path = f"BinarylassOneOnOne\\cp_binary_fault_{vs_label}.ckpt.keras"

        train_round = train_data.copy()

        train_round = train_round[(train_round['faultNumber'] == 0) | (train_round['faultNumber'] == vs_label)]

        # convert to binary labels
        train_round.loc[train_round['faultNumber'] > 1, 'faultNumber'] = 1


        train_X = train_round.iloc[:, 3:].values

        # labels
        train_Y = train_round['faultNumber'].values

        # One hot encode
        pred_Y = model.predict(test_X)
        predictions[str(vs_label)] = pred_Y[:, 0]
        print(pred_Y)
        pred_y = tf.round(pred_Y)  # Convert predictions to binary (0 or 1)

        cm = confusion_matrix(test_Y, pred_Y)

        class_0_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        class_1_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        print("Class 0 Accuracy:", class_0_accuracy)
        print("Class 1 Accuracy:", class_1_accuracy)

        #dat = pd.DataFrame({vs_label: prediction[:, 0]})


    predictions.to_csv('.\\BinarylassOneOnOne\\predictions.csv')

    test_round = test_data.copy()
    test_round = test_round[(test_round['faultNumber'] == 0) | (test_round['faultNumber'] == vs_label)]
    test_round.loc[test_round['faultNumber'] > 1, 'faultNumber'] = 1
    test_Y = test_round['faultNumber'].values
    enc.fit(test_Y.reshape(-1, 1))
    test_labels_enc = enc.transform(test_Y.reshape(-1, 1))