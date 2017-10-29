#!/usr/bin/env python

import sys, socket, select
import json
import numpy as np
from datetime import datetime
import os
import librosa
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils import shuffle
import keras
import sklearn
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten, LSTM, Reshape, Permute, GRU
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D as Max2D
from keras.optimizers import SGD
from keras.models import load_model

import tensorflow as tf

LABEL_BABY_CRY = 0
LABEL_CAR = 1

# About mfcc feature extraction:
# The audio file is first cut into overlapping frames, each frame typically
# is a 20ms clip. For each frame, it extracts 'height' number of mfcc
# coefficient. The 'width' means for each mfcc coefficient, how many sample
# value it has extracted, which typically corresponds to how long the audio
# clip is.
def extract_mfcc(filename, order=2, height=13, width=None):
    print "processing", filename
    y, sr = librosa.load(filename)
    # print filename, "sample rate:", sr
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=height)
    # print "mfcc shape:", mfcc.shape

    mfccs = mfcc
    if order >= 1:
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfccs = np.concatenate((mfccs, mfcc_delta))
    if order >= 2:
        mfcc_delta = librosa.feature.delta(mfcc, order=2)
        mfccs = np.concatenate((mfccs, mfcc_delta))

    print "mfccs.shape:", mfccs.shape
    return mfccs

# Just extract the mfcc features, we will normalize the width a bit later.
def prepare_data(data_dir):
    data = []
    for top, dirs, files in os.walk(data_dir):
        wave_files = [f for f in files if f.endswith(".wav")]
        for f in wave_files:
            fname  = os.path.join(top, f)
            # print fname
            mfcc = extract_mfcc(fname)
            data.append(mfcc)
    # the data need to be same width
    max_width = -1
    for mfcc in data:
        if max_width < mfcc.shape[1]:
            max_width = mfcc.shape[1]
    print "max_width:", max_width
    # pad the mfcc features to max width
    data2 = []
    for mfcc in data:
        if max_width > mfcc.shape[1]:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_width - mfcc.shape[1])), mode='constant', constant_values=0)
        data2.append(mfcc)
    for mfcc in data2:
        print "mfcc.shape:", mfcc.shape
    return np.array(data2)

def train_model():
    data_set = {}
    data_set[LABEL_BABY_CRY] = prepare_data("../data/baby_cry")
    data_set[LABEL_CAR] = prepare_data("../data/car")

    # TODO: might need to normalize width between different classes
    x_data = []
    y_data = []
    for k in data_set.keys():
        for d in data_set[k]:
            x_data.append(d)
            y_data.append(k)
    # the data need to be same width
    width = -1
    for mfcc in x_data:
        if width < mfcc.shape[1]:
            width = mfcc.shape[1]
    print "width:", width
    # pad the mfcc features to mwidth
    x_data2 = []
    for mfcc in x_data:
        if width > mfcc.shape[1]:
            mfcc = np.pad(mfcc, ((0, 0), (0, width - mfcc.shape[1])), mode='constant', constant_values=0)
        x_data2.append(mfcc)
    x_data = x_data2
    for mfcc in x_data:
        print "mfcc.shape:", mfcc.shape

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    sklearn.utils.shuffle(x_data, y_data)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1);

    test_size = int(x_data.shape[0] * 0.2)
    print "total samples:", x_data.shape[0], "test_size:", test_size
    x_test = x_data[0:test_size, ]
    y_test = y_data[0:test_size, ]
    x_train = x_data[test_size:, ]
    y_train = y_data[test_size:, ]

    Y_train = keras.utils.to_categorical(y_train, num_classes=2)
    Y_test = keras.utils.to_categorical(y_test, num_classes=2)

    # Just build a model, any model, for the time being.
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(39, width, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Max2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    adam=keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print model.summary()
    print model.output_shape

    filepath = "weights.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list=[checkpoint]

    print "x_train.shape:", x_train.shape
    print "Y_train", Y_train.shape
    model.fit(x_train, Y_train, batch_size=32, epochs=10, callbacks=callbacks_list,
              verbose=1, validation_split=0.2)

    # model.fit(x_train, Y_train, epochs=20, batch_size=3)

    score = model.evaluate(x_test, Y_test, verbose = 0)
    print "test score:", score

    bestmodel = load_model('weights.best.hdf5')
    bestscore = bestmodel.evaluate(x_test, Y_test, verbose = 0)
    print "best model test score:", score

if __name__ == "__main__":
    train_model()