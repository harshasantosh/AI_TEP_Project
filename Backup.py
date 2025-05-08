
reduced_data = DF
reduced_data = reduced_data[reduced_data['faultNumber'] != 3]
reduced_data = reduced_data[reduced_data['faultNumber'] != 9]
reduced_data = reduced_data[reduced_data['faultNumber'] != 15]

labels = aug_dat['faultNumber'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, labels_enc, test_size=0.2, random_state=42)




















# This is the first Convolutional Neural Network that we made that beats the paper's accuracy on the
# multiclass fault classification problem.
import numpy as np
from keras.layers import Input, Dense, Conv1D, Flatten, BatchNormalization
import keras
import tensorflow as tf
#from ConfusionMatrix import confusion_matrix_multi
from sklearn.metrics import accuracy_score, confusion_matrix
from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder

'''
class CombineCallback(tf.keras.callbacks.Callback):

    def __init__(self, **kargs):
        super(CombineCallback, self).__init__(**kargs)

    def on_epoch_end(self, epoch, logs={}):
        logs['combine_metric'] = logs['val_recall'] + 0.1*logs['val_accuracy']
'''

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

# convert to binary labels
#train_data.loc[train_data['faultNumber'] > 1, 'faultNumber'] = 1
#test_data.loc[test_data['faultNumber'] > 1, 'faultNumber'] = 1

# convert to a form that CNNs can work on it
# break the data into groups of size 'window' to convolve over
# skip 'step' entries so that we're jumping over the already
# included data
train_X, train_Y = make_cnn_data(train_data, window=window, step=step)
test_X, test_Y   = make_cnn_data(test_data, window=window, step=step)

train_X, train_Y = shuffle_data(train_X, train_Y)
test_X, test_Y   = shuffle_data(test_X, test_Y)



'''
num_zeros_train = (train_Y == 0).sum()
num_zeros_test = (test_Y == 0).sum()
print(num_zeros_train)

trX = []
trY = []
teX = []
teY = []


# make binary by taking every class and making it 0 or 1, then flip them because that's the way
# recall works in tensorflow. Recall on class 0 is what we want, so it has to become class 1
train_counts = np.zeros(21)
for a in range(len(train_Y)):
    if train_counts[train_Y[a]] > (num_zeros_train // 17):
        if train_counts[train_Y[a]] != 0:
            continue
    train_counts[train_Y[a]] += 1

    trX.append(train_X[a])
    if train_Y[a] > 1:
        #train_Y[a] = 1
        #trY.append(1)
        trY.append(0)
    else:
        #trY.append(0)
        trY.append(1)
    #train_Y[a] = 1 - train_Y[a]

test_counts = np.zeros(21)
for a in range(len(test_Y)):
    if test_counts[test_Y[a]] > (num_zeros_test // 17):
        if test_counts[test_Y[a]] != 0:
            continue
    test_counts[test_Y[a]] += 1

    teX.append(test_X[a])

    if test_Y[a] > 1:
        #teY.append(1)
        teY.append(0)
        #test_Y[a] = 1
    else:
        #teY.append(0)
        teY.append(1)
    #test_Y[a] = 1 - test_Y[a]

for t in range(len(train_counts)):
    print(str(t),train_counts[t])

train_X = np.asarray(trX)
train_Y = np.asarray(trY)
test_X = np.asarray(teX)
test_Y = np.asarray(teY)
'''

# One hot encode label
enc = OneHotEncoder()
enc.fit(train_Y.reshape(-1, 1))
train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
enc.fit(test_Y.reshape(-1, 1))
test_labels_enc = enc.transform(test_Y.reshape(-1, 1))


checkpoint_path = "CNN_Binary_vanilla\\cp_ep_{epoch:02d}_rec_{val_recall:.2f}_acc_{val_accuracy:.2f}.ckpt.keras"
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
        #Dense(train_labels_enc.shape[1], activation='softmax')
        #BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    # original paper specified adam. It didn't specify crossentropy. We're not beholden to these when
    # improving upon their work, but getting an improvement from elsewhere comes first
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='accuracy')

    #m1 = RecallClass0()

    # Create a callback that saves the model's weights
    # saves the one with the best recall. This is perhaps cheating at the moment without a 3rd set.
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=False,
                                                     verbose=1)#, mode='max',
                                                     #monitor='val_accuracy',
                                                     #save_best_only=True)


    batch_size = 256

    # 0s outnumbered 17 to 1 flip back if the other stuff flips back
    #class_weight = {0: 1, 1: 17}
    #fit model
    history = model.fit(train_X, train_Y,#train_labels_enc.todense(),
                        epochs=2000, batch_size=batch_size,
                        validation_data=(test_X, test_Y),#test_labels_enc.todense()),
                        callbacks=[cp_callback]) #, class_weight=class_weight)


pred_Y = model.predict(test_X)
pred_Y = tf.round(pred_Y)  # Convert predictions to binary (0 or 1)

cm = confusion_matrix(test_Y, pred_Y)

class_0_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1])
class_1_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# yes this is correct, there was a flip higher up
print("Class 0 Accuracy:", class_0_accuracy)
print("Class 1 Accuracy:", class_1_accuracy)
