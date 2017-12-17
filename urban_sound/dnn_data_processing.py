# ==================================================
# Copyright (C) 2017-2017
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 12/11/2017
#
# This file is part of UrbanSoundClassification project.
# 
# This can not be copied and/or distributed 
# without the express permission of yilin.shen
# ==================================================

import glob
import os
import librosa
import pickle
import numpy as np


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    return mfccs, chroma, mel, contrast


def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav'):
    # features, labels = np.empty((0, 193)), np.empty(0)
    features, labels = np.empty((0, 187)), np.empty(0)

    for label, sub_dir in enumerate(sub_dirs):
        print("Processing folder [%s]..." % sub_dir)

        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs, chroma, mel, contrast = extract_feature(fn)
            ext_features = np.hstack([mfccs, chroma, mel, contrast])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, fn.split('-')[1])

    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1

    return one_hot_encode


def data_preparation(data_dir, class_dir):
    features, labels = parse_audio_files(data_dir, class_dir)
    labels = one_hot_encode(labels)

    return features, labels


def main():
    # extract features and labels
    features, labels = data_preparation('../UrbanSound8K/audio',
                                        ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10'])

    # save feature and label files
    pickle.dump(features, open("../UrbanSound8K/audio/dnn_features.p", "wb"))
    pickle.dump(labels, open("../UrbanSound8K/audio/dnn_labels.p", "wb"))

    # print("Print features:")
    # print(features)
    # print("\nPrint labels:")
    # print(labels)


if __name__ == '__main__':
    main()
