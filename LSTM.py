




from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout
from keras.optimizers import Adam
import tensorflow as tf
from ConfusionMatrix import confusion_matrix_multi
from sklearn.metrics import accuracy_score

from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder

# original dataset
unaug_dat = read_rdata('C:\\Users\\Arthur\\PycharmProjects\\AI_Poject\\TEP_FaultFree_Training.RData',
           'C:\\Users\\Arthur\\PycharmProjects\\AI_Poject\\TEP_Faulty_Training.RData')

# no 3 9 or 15
# also no datapoints before sample = 20 because fault not introduced yet
unaug_dat = remove_bad_faults(unaug_dat)

# min max scale as in original paper
unaug_dat = data_scaler(unaug_dat)

# create multiple channels for data before and after the current datapoint which is in the middle


window = 10
step = 20

# as in original paper, split 60% training 40% testing
train_data, test_data = split_dataset(unaug_dat)

# convert to a form that CNNs can work on it
train_X, train_Y = make_cnn_data(train_data, window=window, step=step)
test_X, test_Y   = make_cnn_data(test_data, window=window, step=step)

# One hot encode label
enc = OneHotEncoder()
enc.fit(train_Y.reshape(-1, 1))
train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
enc.fit(test_Y.reshape(-1, 1))
test_labels_enc = enc.transform(test_Y.reshape(-1, 1))

# Input is of size 52
checkpoint_path = "LSTM\\cp.ckpt.keras"
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
        Input(shape=(train_X.shape[1],train_X.shape[2])),
        Bidirectional(LSTM(128, activation='gelu', return_sequences=True)),
        LSTM(128, activation='gelu'),
        Dropout(0.5),
        Dense(400, activation='gelu'),
        #Dense(40, activation='gelu'),
        Dense(train_labels_enc.shape[1], activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.00001)

    # original paper specified adam. It didn't specify crossentropy, but it didn't specify otherwise
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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

