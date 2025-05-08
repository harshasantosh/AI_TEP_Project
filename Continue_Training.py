from keras.layers import Input, Dense, Conv1D, Flatten, Dropout
import keras
import tensorflow as tf
from ConfusionMatrix import confusion_matrix_multi, histogram
from sklearn.metrics import accuracy_score
from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder
import json
import os
if not os.path.exists('./history_multiclass'):
    os.makedirs('./history_multiclass')

# original dataset
unaug_dat = read_rdata('.\\TEP_FaultFree_Training.RData',
                       '.\\TEP_Faulty_Training.RData')

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1)

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

# convert to a form that CNNs can work on it


# One hot encode label


#for step in [1,10,20]:
window = 15
step = 1
train_X, train_Y = make_cnn_data(train_data, window=window, step=step)
test_X, test_Y = make_cnn_data(test_data, window=window, step=step)

enc = OneHotEncoder()
enc.fit(train_Y.reshape(-1, 1))
train_labels_enc = enc.transform(train_Y.reshape(-1, 1))
enc.fit(test_Y.reshape(-1, 1))
test_labels_enc = enc.transform(test_Y.reshape(-1, 1))

#input_model = '.\\CNN_Multi_Hyper\\cp_15_1_tanh_kern_3_opt_adam_drop_0.ckpt.keras'
checkpoint_path = '.\\Continued_Model\\cp_15_1_tanh_kern_3_opt_adam_drop_0.ckpt.keras'
model = tf.keras.models.load_model(checkpoint_path)

prediction = enc.inverse_transform(model.predict(test_X, verbose=0))
true = enc.inverse_transform(test_labels_enc)

print("Accuracy:", accuracy_score(true, prediction))

# Confusion matrix to show accuracies
#confusion_matrix_multi(true, prediction, 'Confusion Matrix')
histogram(true, prediction, 'Class Accuracy Histogram')


'''
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                         save_weights_only=False,
                                                                         verbose=1)

batch_size = 256
# fit model
history = model.fit(train_X, train_labels_enc.todense(), epochs=100, batch_size=batch_size,
                    validation_data=(test_X, test_labels_enc.todense()),
                    callbacks=[cp_callback])

with open(f'.\\Continued_Model\\history_15_1_tanh_kern_3_opt_adam_drop_0.ckpt.keras', 'w') as f:
    json.dump(history.history, f)
'''