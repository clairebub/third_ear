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
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten, LSTM, Reshape, Permute, GRU
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D as Max2D
from keras.utils import np_utils
from keras.models import load_model
import tensorflow as tf

# About mfcc feature extraction:
# The audio file is first cut into overlapping frames, each frame typically
# is a 20ms clip. For each frame, it extracts 'height' number of mfcc
# coefficient. The 'width' means for each mfcc coefficient, how many sample
# value it has extracted, which typically corresponds to how long the audio
# clip is.
def extract_mfcc(filename, order=2, height=13, width=None):
    y, sr = librosa.load(filename)
    print filename, "sample rate:", sr
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=height)
    print "mfcc shape:", mfcc.shape

    mfccs = mfcc
    if order >= 1:
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfccs = np.concatenate((mfccs, mfcc_delta))
    if order >= 2:
        mfcc_delta = librosa.feature.delta(mfcc, order=2)
        mfccs = np.concatenate((mfccs, mfcc_delta))
    print "order:", order, "mfccs.shape:", mfccs.shape

    if width and width > mfccs.shape[1]:
        mfccs = np.pad(mfccs, ((0, 0), (0, width - mfccs.shape[1])), mode='constant', constant_values=0)
    return mfccs

# Just extract the mfcc features, we will normalize the width a bit later.
def prepare_base_data(data_dir):
    data = []
    for top, dirs, files in os.walk(data_dir):
        for f in files:
            fname  = os.path.join(top, f)
            mfcc = extract_mfcc(fname)
            data.append(mfcc)
    # the data need to be same width
    for mfcc in data:
        print "width:", mfff.shape[1]




if __name__ == "__main__":
    prepare_base_data("../../", )
#    extract_mfcc("speech.wav")
 #   extract_mfcc("0963.wav")