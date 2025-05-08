import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from DataProcessing import *
from sklearn.preprocessing import OneHotEncoder

# original dataset
unaug_dat = read_rdata('C:\\Users\\Arthur\\PycharmProjects\\AI_Poject\\TEP_FaultFree_Training.RData',
           'C:\\Users\\Arthur\\PycharmProjects\\AI_Poject\\TEP_Faulty_Training.RData')

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
inputs = Input(shape=(train_X.shape[1],))

# the network is described as 102-102-50-40-18
# this description makes no sense, either it's 52-102-102-50-40-18 or 102-102-50-40 with the 52 and 18 implied
# this is what we implemented
hidden_layer = Dense(102, activation='relu')(inputs)
hidden_layer = Dense(102, activation='relu')(hidden_layer)
hidden_layer = Dense(50, activation='relu')(hidden_layer)
hidden_layer = Dense(40, activation='relu')(hidden_layer)

outputs = Dense(train_labels_enc.shape[1], activation='softmax')(hidden_layer)

# original paper specified adam. It didn't specify crossentropy, but it didn't specify otherwise
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#print(train_data.shape[0]// 50)
#batch_size = train_data.shape[0] // 50

batch_size = 256

#
history = model.fit(train_X, train_labels_enc.todense(), epochs=400, batch_size=batch_size, validation_data=(test_X, test_labels_enc.todense()))

# Plot the training history for loss and accuracy
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# Import the required libraries
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


# Create a function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    f, ax = plt.subplots(figsize=(15, 15))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_ylim(len(cm), 0)
    plt.tight_layout()
    plt.show()


y_pred = enc.inverse_transform(model.predict(test_X, verbose=0))
y_true = enc.inverse_transform(test_Y)

# Plot the confusion matrix and print the f1 score for each algorithm
plot_confusion_matrix(y_true, y_pred, 'Neural Net Confusion Matrix')
print("Neural Net accuracy_score:", accuracy_score(y_true, y_pred))
