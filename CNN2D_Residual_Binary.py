# This is the first Convolutional Neural Network that we made that beats the paper's accuracy on the
# multiclass fault classification problem.

from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, BatchNormalization, Activation, Add
import keras
import tensorflow as tf
from ConfusionMatrix import confusion_matrix_multi
from sklearn.metrics import accuracy_score, confusion_matrix
from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# read original dataset
unaug_dat = read_rdata('./data/TEP_FaultFree_Training.RData',
           './data/TEP_Faulty_Training.RData')

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
train_X, train_Y = make_cnn_data(train_data, window=window, step=step)
test_X, test_Y   = make_cnn_data(test_data, window=window, step=step)

# make binary by taking every class and making it 0 or 1, then flip them because that's the way
# recall works in tensorflow. Recall on class 0 is what we want, so it has to become class 1
for a in range(len(train_Y)):
    if train_Y[a] > 1:
        train_Y[a] = 1
    #train_Y[a] = 1 - train_Y[a]

for a in range(len(test_Y)):
    if test_Y[a] > 1:
        test_Y[a] = 1
    #test_Y[a] = 1 - test_Y[a]

# One hot encode label
enc = OneHotEncoder()
enc.fit(train_Y.reshape(-1, 1))
train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
enc.fit(test_Y.reshape(-1, 1))
test_labels_enc = enc.transform(test_Y.reshape(-1, 1))


checkpoint_path = "./CNN2D_Residual_Binary/cp_17_to_1_epochs6000_dropouts2_1_res_layer_k5_w10_s20.ckpt.keras"
load = False
if load:
    model = tf.keras.models.load_model(checkpoint_path)
else:
    # All activations (except the final one) are gelu
    # Inputs are reshaped to be in a groups of size 'window'
    # A 1D convolutional layer with kernel size 5 and
    #   64 filters goes over this input
    # This is passed to another Conv2D layer with the same setup
    # This is then flattened to use in dense layers
    # This flattened data is fed into the same 50 / 40 dense layers that the paper used
    # Finally the output is an 18 neuron softmax
    '''
    model = tf.keras.Sequential([
        Input(shape=(train_X.shape[1], train_X.shape[2], 1)),
        Conv2D(filters=64, kernel_size=5, activation='gelu'),
        Dropout(.2),
        Conv2D(filters=64, kernel_size=5, activation='gelu'),
        Flatten(),
        Dropout(.2),
        Dense(50, activation='gelu'),
        Dropout(.2),
        Dense(40, activation='gelu'),
        Dense(train_labels_enc.shape[1], activation='softmax')
    ])
    '''


    def residual_block(x, filters, kernel_size=3):
        shortcut = x

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        x = Dropout(.2)(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)

        x = Add()([x, shortcut])
        #x = Dropout(.2)(x)
        x = Activation('gelu')(x)
        return x


    input_layer = Input(shape=(train_X.shape[1], train_X.shape[2], 1))
    x = Conv2D(64, 7, padding='same', activation='gelu')(input_layer)
    x = Dropout(.2)(x)
    x = residual_block(x, 64, 5)
    #x = residual_block(x, 64, 3)
    x = Flatten()(x)
    x = Dropout(.2)(x)
    x = Dense(64, activation='gelu')(x)
    x = Dropout(.2)(x)
    x = Dense(64, activation='gelu')(x)
    output_layer = Dense(1, activation='sigmoid')(x) #Dense(train_labels_enc.shape[1], activation='softmax')(x)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model = tf.keras.Model(input_layer, output_layer)

    # original paper specified adam. It didn't specify crossentropy. We're not beholden to these when
    # improving upon their work, but getting an improvement from elsewhere comes first
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'recall'])

    # m1 = RecallClass0()

    # Create a callback that saves the model's weights
    # saves the one with the best recall. This is perhaps cheating at the moment without a 3rd set.
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=False,
                                                     verbose=1)

    batch_size = 256

    # 0s outnumbered 17 to 1 flip back if the other stuff flips back
    class_weight = {0: 17, 1: 1}
    # fit model
    history = model.fit(train_X, train_Y,  # train_labels_enc.todense(),
                        epochs=6000, batch_size=batch_size,
                        validation_data=(test_X, test_Y),  # test_labels_enc.todense()),
                        callbacks=[cp_callback], class_weight=class_weight)


pred_Y = model.predict(test_X)
pred_Y = tf.round(pred_Y)  # Convert predictions to binary (0 or 1)

cm = confusion_matrix(test_Y, pred_Y)

class_0_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1])
class_1_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# yes this is correct, there was a flip higher up
print("Class 0 Accuracy:", class_0_accuracy)
print("Class 1 Accuracy:", class_1_accuracy)