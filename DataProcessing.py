import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pyreadr
import tensorflow as tf


def remove_bad_faults(data):
    data = data[data['faultNumber'] != 3]
    data = data[data['faultNumber'] != 9]
    data = data[data['faultNumber'] != 15]

    data = data.loc[data['sample'] > 20]

    return data


# original data is in the form of rdata files
def read_rdata(faultFree, faulty):
    FaultFree = pyreadr.read_r(faultFree)['fault_free_training']
    Faulty = pyreadr.read_r(faulty)[ 'faulty_training']
    return pd.concat([FaultFree, Faulty])


# in the binary classifier case, the data is too lopsided, it needs to be made more equal
# or else the model guesses 1 pracitically every time. Only really needed for the trianing
# set
def balance_data(dataset):
    '''
    num_faults = len(dataset['faultNumber'].unique)

    balance_dataset = dataset[dataset['faultNumber'] == 0]
    num_zeros = balance_dataset.shape[0]

    for x in range(num_faults):
        balance_dataset = pd.concat([balance_dataset, dataset[dataset['faultNumber'] == x].sample(num_zeros/(num_faults-1))])
    '''
    zeros = dataset[dataset['faultNumber'] == 0]
    ones  = dataset[dataset['faultNumber'] != 0]

    for x in range(len(dataset['faultNumber'].unique())):
        ones = pd.concat([zeros, ones])

    return ones

# split the data in the manner of the original paper
def split_dataset(dataset, percent=.6):
    # the original paper split the data by using simulations 1 through 300 to train and 301 to 500 for each
    # fault type
    split = int(max(dataset['simulationRun'].unique())*percent)
    train_dataset = dataset.loc[(dataset['simulationRun'] <= split)] #indexed at 1
    test_dataset  = dataset.loc[(dataset['simulationRun'] > split)]

    # first hour nothing happens
    #train_dataset = train_dataset.loc[train_dataset['sample'] > 20]
    #test_dataset = test_dataset.loc[test_dataset['sample'] > 20]

    # randomize order *new, check*
    # train_dataset = train_dataset.sample(frac=1)
    # test_dataset = test_dataset.sample(frac=1)

    return train_dataset, test_dataset


# 'mimics dpca'. only confirmed to work for window 3 atm... need to fix for window 2 maybe
# mimicking dpca seems to mean adding consecutive examples
def combine_dataset(data, window):
    temp_data = []
    for x in range(window):
        # Create window size copies of the dataset offset by x with their own internal indices
        # x goes from 0 to window size, so if it's 3 say there are three sets that each start from 0, 1, 2
        temp = data.iloc[x:]
        temp.reset_index(drop=True, inplace=True) #reset the index to allow for the next step

        # each one grabs groups of size window and sums them. the off by xs up to x = window-1
        # ensure that the whole set is covered
        temp = temp.groupby(temp.index // window).sum()
        temp_data.append(temp)

    # this combines them back into one dataset and in order
    result = pd.concat(temp_data).sort_index(kind='merge')
    # reset indices
    result.reset_index(drop=True, inplace=True)

    # fault number being added is not intentional, revert. Also flags undesirable bleed over between faults
    # by making them not be integers
    result['faultNumber'] = result['faultNumber'].div(window)

    # to get rid of examples that straddle categories, remove all that don't have integers for fault numbers
    # for up to window = 3 this works. Window = 4 it doesn't because of fault numbers being skipped in the
    # paper's implementation. It would work if this was done prior to fault numbers being skipped though.
    result_final = result[result['faultNumber'] == result['faultNumber'].astype(int)]
    result_final = result_final[result_final['faultNumber'] != 3]
    result_final = result_final[result_final['faultNumber'] != 9]
    result_final = result_final[result_final['faultNumber'] != 15]
    if result_final.iloc[-1]['faultNumber'] != 20:
        result_final = result_final[:-1]


    # original paper 0-1 scales
    result_final = data_scaler(result_final)

    return result_final


# the original paper 01 scales the data
def data_scaler(data):
    # Columns to scale
    columns_to_scale = data.columns[4:]

    # Create a ColumnTransformer
    ct = ColumnTransformer(
        transformers=[('scaler', MinMaxScaler(), columns_to_scale)],
        remainder='passthrough'  # Keep the remaining columns as they are
    )

    # Fit and transform the data
    data_scaled = ct.fit_transform(data)

    # Convert the result back to a DataFrame (optional)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    #Column Transformer messes up column order
    cols = data_scaled.columns.to_list()
    cols = cols[-4:] + cols[0:-4]
    data_scaled = data_scaled[cols]

    #reset column names
    data_scaled.columns = data.columns

    #round to cut down on space
    data_scaled = data_scaled.round(5)

    return data_scaled


def use_window(data, window, step, faultNumber):
    features_total = []
    labels_total = []
    for i in range(0, len(data) - window, step):
        features = np.array(data.iloc[i:i + window, 3:])
        #labels = np.array(data.iloc[i + window - 1, 0])

        labels = np.array(faultNumber)
        features_total.append(features)
        labels_total.append(labels)

    return np.array(features_total), np.array(labels_total)

def make_cnn_data(data, window=10, step=20, flip=False):
    cnn_features = []
    cnn_labels = []

    classes = int(data['faultNumber'].max() + 1)


    for faultNumber in range(classes):
        if faultNumber == 3 or faultNumber == 9 or faultNumber == 15:
            continue

        # train is typically from 0 to 300 and test is typically 301 to 500 or so
        for simulationRun in range(int(data['simulationRun'].min()), int(data['simulationRun'].max())):
            # contain to a fault number and a simulation run, drop all the stuff until you hit 20 because that's when the
            # fault is introduced
            d = data[
                (data['faultNumber'] == faultNumber) & (data['simulationRun'] == simulationRun) & (
                            data['sample'] > 20)]

            cnn_feature, cnn_label = use_window(d, window, step, faultNumber)
            if flip:
                cnn_features.append(np.transpose(cnn_feature))
            else:
                cnn_features.append(cnn_feature)
            cnn_labels.append(cnn_label)

    return np.array(cnn_features).reshape((-1, window, 52)), \
        np.array(cnn_labels).reshape((-1))


def shuffle_data(data, label):
    # Create an index array
    indices = np.arange(data.shape[0])

    # Shuffle the index array
    np.random.shuffle(indices)

    # Use the shuffled indices to reorder your data
    data = data[indices]
    label = label[indices]

    return data, label

# Create a custom metric for recall on class 0
#class RecallClass0(tf.keras.metrics.Recall):
#    def __init__(self, name='recall_class_0', **kwargs):
#        super().__init__(name=name, **kwargs)
#
#    def update_state(self, y_true, y_pred, sample_weight=None):
#        y_true = tf.cast(y_true, tf.int32)
#        y_pred = tf.cast(y_pred, tf.int32)
#
#        # Only consider class 0
#        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(y_true == 0, y_pred == 0), tf.float32))
#        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(y_true == 0, y_pred != 0), tf.float32))
#
#        super().update_state(true_positives, false_negatives, sample_weight)

# Define recall metric with correct type casting and thresholding
def recall(y_true, y_pred):
    # Ensure y_true is of type float32 (if needed)
    y_true = tf.cast(y_true, tf.int64)# tf.float32)

    # Convert probabilities to binary outcomes (0 or 1) using threshold
    y_pred_binary = tf.cast(y_pred > 0.5, tf.int64)# tf.float32)  # Threshold at 0.5

    # Calculate true positives and false negatives
    true_positives = tf.reduce_sum(y_true * y_pred_binary)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred_binary))

    # Compute recall
    recall_value = true_positives / (true_positives + false_negatives)# + tf.keras.backend.epsilon())
    return recall_value


