import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv1D, Flatten, Dropout
from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from ConfusionMatrix import confusion_matrix_multi, histogram
import gc
import os

# original dataset
unaug_dat = read_rdata('.\\TEP_FaultFree_Training.RData',
                       '.\\TEP_Faulty_Training.RData')


unaug_dat = remove_bad_faults(unaug_dat)

# min max scale as in original paper
unaug_dat = data_scaler(unaug_dat)

train_data, test_data = split_dataset(unaug_dat)
del unaug_dat
del train_data
gc.collect()


window = 15
step = 1
test_X, test_Y = make_cnn_data(test_data, window=window, step=step)

enc = OneHotEncoder()
enc.fit(test_Y.reshape(-1, 1))
test_labels_enc = enc.transform(test_Y.reshape(-1, 1))


#paths = ['.\\selu3\\cp_15_1_selu_kern_3_opt_adam_drop_0.ckpt.keras', '.\\tanh3\\cp_15_1_tanh_kern_3_opt_adam_drop_0.ckpt.keras',
#    '.\\CNN_Multi_Hyper\\cp_15_1_tanh_kern_5_opt_adam_drop_0.ckpt.keras',
#         '.\\Continued_Model\\cp_15_1_tanh_kern_3_opt_adam_drop_0.ckpt.keras']
paths = ['H:\\Hisotries\\cp_15_1_leaky_relu_kern_5_opt_adam_drop_0.ckpt.keras']

for path in paths:
    print(path)
    model = tf.keras.models.load_model(path)
    prediction = enc.inverse_transform(model.predict(test_X))

    true = enc.inverse_transform(test_labels_enc)

    print("Accuracy:", accuracy_score(true, prediction))

    histogram(true, prediction, 'Class Accuracy Histogram')


    bin_true = []
    bin_pred = []

    for t in true:
        if t > 0:
            bin_true.append(1)
        else:
            bin_true.append(0)

    for p in prediction:
        if p > 0:
            bin_pred.append(1)
        else:
            bin_pred.append(0)


    print("Accuracy:", accuracy_score(bin_true, bin_pred))