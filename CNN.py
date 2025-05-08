# This is the first Convolutional Neural Network that we made that beats the paper's accuracy on the
# multiclass fault classification problem.

from keras.layers import Input, Dense, Conv1D, Flatten
import keras
import tensorflow as tf
from ConfusionMatrix import confusion_matrix_multi
from sklearn.metrics import accuracy_score
from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder

# read original dataset
unaug_dat = read_rdata('C:\\Users\\Arthur\\PycharmProjects\\AI_Poject\\TEP_FaultFree_Training.RData',
           'C:\\Users\\Arthur\\PycharmProjects\\AI_Poject\\TEP_Faulty_Training.RData')

# no 3 9 or 15
# also no datapoints before sample = 20 because fault not introduced yet
unaug_dat = remove_bad_faults(unaug_dat)

# min max scale as in original paper
unaug_dat = data_scaler(unaug_dat)

window = 10
step = 20

# as in original paper, split 60% training 40% testing
train_data, test_data = split_dataset(unaug_dat)

# convert to a form that CNNs can work on it
# break the data into groups of size 'window' to convolve over
# skip 'step' entries so that we're jumping over the already
# included data
train_X, train_Y = make_cnn_data(train_data, window=window, step=step) #, flip=True)
test_X, test_Y   = make_cnn_data(test_data, window=window, step=step) #, flip=True)

# One hot encode label
enc = OneHotEncoder()
enc.fit(train_Y.reshape(-1, 1))
train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
enc.fit(test_Y.reshape(-1, 1))
test_labels_enc = enc.transform(test_Y.reshape(-1, 1))


checkpoint_path = "CNN\\cp_new.ckpt.keras"
load = False
if load:
    model = tf.keras.models.load_model(checkpoint_path)
else:
    # All activations (except the final one) are gelu
    # Inputs are reshaped to be in a groups of size 'window'
    # A 1D convolutional layer with kernel size 5 and
    #   64 filters goes over this input
    # This is passed to another Conv1D layer with the same setup
    # This is then flattened to use in dense layers
    # This flattened data is fed into the same 50 / 40 dense layers that the paper used
    # Finally the output is an 18 neuron softmax
    model = tf.keras.Sequential([
        Input(shape=(train_X.shape[1], train_X.shape[2])),
        Conv1D(filters=64, kernel_size=5, activation='gelu'),
        Conv1D(filters=64, kernel_size=5, activation='gelu'),
        Flatten(),
        Dense(50, activation='gelu'),
        Dense(40, activation='gelu'),
        Dense(train_labels_enc.shape[1], activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0005)

    # original paper specified adam. It didn't specify crossentropy. We're not beholden to these when
    # improving upon their work, but getting an improvement from elsewhere comes first
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=False,
                                                     verbose=1)


    batch_size = 256
    #fit model
    history = model.fit(train_X, train_labels_enc.todense(), epochs=600, batch_size=batch_size,
                        validation_data=(test_X, test_labels_enc.todense()),
                        callbacks=[cp_callback])


prediction = enc.inverse_transform(model.predict(test_X, verbose=0))
true = enc.inverse_transform(test_labels_enc)

print("Accuracy:", accuracy_score(true, prediction))

# Confusion matrix to show accuracies
confusion_matrix_multi(true, prediction, 'Confusion Matrix')
