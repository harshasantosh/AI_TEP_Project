from keras.layers import Input, Dense, Conv1D, Flatten, Dropout
import keras
import tensorflow as tf
from ConfusionMatrix import confusion_matrix_multi
from sklearn.metrics import accuracy_score
from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder
import json

# original dataset
unaug_dat = read_rdata('./data/TEP_FaultFree_Training.RData',
                       './data/TEP_Faulty_Training.RData')

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

for window in [5, 10, 15]:
    for step in [1,10,20,30]:
        train_X, train_Y = make_cnn_data(train_data, window=window, step=step)
        test_X, test_Y = make_cnn_data(test_data, window=window, step=step)

        # make binary by taking every class and making it 0 or 1, then flip them because that's the way
        # recall works in tensorflow. Recall on class 0 is what we want, so it has to become class 1
        for a in range(len(train_Y)):
            if train_Y[a] > 1:
                train_Y[a] = 1
            # train_Y[a] = 1 - train_Y[a]

        for a in range(len(test_Y)):
            if test_Y[a] > 1:
                test_Y[a] = 1
            # test_Y[a] = 1 - test_Y[a]

        enc = OneHotEncoder()
        enc.fit(train_Y.reshape(-1, 1))
        train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
        enc.fit(test_Y.reshape(-1, 1))
        test_labels_enc = enc.transform(test_Y.reshape(-1, 1))

        for drop in [0, .2, .5]:
            for act in ['relu', 'gelu', 'selu', 'tanh']:
                for kern in [3, 5, 7]:
                    for opt in ['adam', 'adadelta', 'adamw']:
                        try:
                            checkpoint_path = f"./CNN_Binary_Hyper/cp_{window}_{step}_{act}_kern_{kern}_opt_{opt}_drop{drop}.ckpt.keras"
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
                                    Input(shape=(train_X.shape[1], train_X.shape[2])),
                                    Conv1D(filters=64, kernel_size=kern, activation=act),
                                    Dropout(drop),
                                    Conv1D(filters=64, kernel_size=kern, activation=act),
                                    Flatten(),
                                    Dropout(drop),
                                    Dense(50, activation=act),
                                    Dense(40, activation=act),
                                    Dense(train_labels_enc.shape[1], activation='softmax')
                                ])

                                optimizer = keras.optimizers.Adam(learning_rate=0.0001)
                                if opt == 'adadelta':
                                    optimizer = keras.optimizers.Adadelta(learning_rate=0.0001)
                                if opt == 'adamw':
                                    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
                                # original paper specified adam. It didn't specify crossentropy, but it didn't specify otherwise AdadeltaOptimizer?
                                model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                                              metrics=['accuracy'])

                                # Create a callback that saves the model's weights
                                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                                 save_weights_only=False,
                                                                                 verbose=1)

                                batch_size = 256
                                # fit model
                                history = model.fit(train_X, train_labels_enc.todense(), epochs=600,
                                                    batch_size=batch_size,
                                                    validation_data=(test_X, test_labels_enc.todense()),
                                                    callbacks=[cp_callback])

                                with open(f'history_binary./history_{window}_{step}_{act}_kern_{kern}_opt_{opt}.json', 'w') as f:
                                    json.dump(history.history, f)

                            prediction = enc.inverse_transform(model.predict(test_X, verbose=0))
                            true = enc.inverse_transform(test_labels_enc)

                            print("Accuracy:", accuracy_score(true, prediction))

                            # Confusion matrix to show accuracies
                            confusion_matrix_multi(true, prediction, 'Confusion Matrix')
                        except Exception:
                            print("error in ", window, step, act, kern, opt)
