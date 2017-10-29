#!/usr/bin/env python

import json
import keras
import librosa
import numpy as np
import os
import sklearn
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten, LSTM, Reshape, Permute, GRU
from keras.optimizers import SGD

# About mfcc feature extraction:
# The audio file is first cut into overlapping frames, each frame typically
# is a 20ms clip. For each frame, it extracts 'height' number of mfcc
# coefficient. The 'width' means for each mfcc coefficient, how many sample
# value it has extracted, which typically corresponds to how long the audio
# clip is.
def extract_mfcc(filename, order=2, height=13):
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

def prep_test_data(filename):
    width = 1126
    mfcc = extract_mfcc(filename)
    mfcc = np.pad(mfcc, ((0, 0), (0, width - mfcc.shape[1])), mode='constant', constant_values=0)
    test_data = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
    return test_data

def test_model(filename):
    test_data = prep_test_data(filename)
    print "test_data.shape", test_data.shape

    model = keras.models.load_model('weights.best.hdf5')
    score = model.predict(test_data)[0]
    print score.shape, "score:", score

def easy_test():
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print "eval easy_score:", score
    xx = np.random.random(20).reshape(1, 20)
    score = model.predict(xx)
    print "predict easy_score:", score

if __name__ == "__main__":
    #easy_test()
    # sys.exit(0)

    prog_name = sys.argv[0]
    if len(sys.argv) != 2:
        print "Usage:", prog_name, "input_file"
        sys.exit(1)
    input_file = sys.argv[1]
    test_model(input_file)

