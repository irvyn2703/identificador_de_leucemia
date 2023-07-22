import random
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv
import tensorflow as tf
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

def redNeuronal(capas, neuronas, epocas, lnRate, momento):
    lnRate = lnRate / 1000
    momento = momento / 100

    df_train = pd.read_csv("DatosFinal/Datos_Train.csv")
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values

    df_test = pd.read_csv("DatosFinal/Datos_Test.csv")
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = label_encoder.transform(y_test)
    y_test = to_categorical(y_test, num_classes=10)

    model = Sequential()
    model.add(Dense(neuronas, activation='relu', input_shape=(X_train.shape[1],)))

    for n in range(capas - 2):
        model.add(Dense(neuronas, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    opt = SGD(learning_rate=lnRate, momentum=momento)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='accuracy', patience=110)

    history = model.fit(X_train, y_train, batch_size=1200, epochs=epocas, callbacks=[checkpoint, early_stopping], validation_data=(X_test, y_test))

    accuracy_values_train = history.history['accuracy']
    accuracy_values_test = history.history['val_accuracy']

    return accuracy_values_train, accuracy_values_test

resul_train, resul_test = redNeuronal(6, 16, 66, 9, 75)

# Plotting the training and testing accuracy
epochs = range(1, len(resul_train) + 1)

plt.plot(epochs, resul_train, 'b', label='Training Accuracy')
plt.plot(epochs, resul_test, 'r', label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
