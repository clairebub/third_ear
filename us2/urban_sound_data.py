#!/usr/bin/env python

import glob
import librosa
import numpy as np
import pandas as pd
import os
import soundfile as sf
import sys
import tensorflow as tf

RUN_FEATURE_EXTRACTION = True
PROJECT_ROOT_DIR = "../"
RAW_DATA_DIR = PROJECT_ROOT_DIR + "data/raw"
FEATURE_DATA_DIR = PROJECT_ROOT_DIR + "data/feature"

# we will take 3 seconds audio and look at melspectrogram at some mel frequencies
# in the range of (FREQUENCY_MIN, FREQUENCY_MAX)
DURATION = 3
N_MFCC= 40
FREQUENCY_MIN = 20
FREQUENCY_MAX = 11000
SOUND_CLASSES = {
    0 : 'air_conditioner',
    1 : 'car_horn',
    2 : 'children_playing',
    3 : 'dog_bark',
    4 : 'drilling',
    5 : 'engine_idling',
    6 : 'gun_shot',
    7 : 'jackhammer',
    8 : 'siren',
    9 : 'street_music',
}

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    buffer_size = features.shape[0]
    dataset = dataset.shuffle(buffer_size).repeat().batch(batch_size)
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset

def load_data(train_set_percent=1):
    """Returns the urban sound data set as dataframes.
    As data frames of (train_x, train_y), (test_x, test_y).
    """
    features, labels = np.zeros(0), np.zeros(0, dtype=int)
    num_classes = len(SOUND_CLASSES)
    for dir_label in range(num_classes):
        dir_label = str(dir_label)
        for fn in glob.glob(os.path.join(RAW_DATA_DIR, dir_label, "*.wav")):
            mfcc2 = _extract_features_from_one_file(fn)
            if mfcc2 is None:
                continue
            features = np.append(features, mfcc2)
            labels= np.append(labels, int(dir_label))
    # turn features into DataFrame
    features = features.reshape(-1, N_MFCC)
    col_names = []
    for i in range(N_MFCC):
        col_name = "mfcc_%02d"%i
        col_names.append(col_name)
    features = pd.DataFrame(data = features, columns = col_names)
    # turn labels into DataFrame
    labels = pd.DataFrame(data = labels, dtype=np.int64, columns = ["y"])
    # split into 80:20 for training and test
    num_training_examples = int(features.shape[0] * train_set_percent)
    train_x = features[:num_training_examples]
    train_y = labels[:num_training_examples]
    test_x = features[num_training_examples:]
    test_y = labels[num_training_examples:]
    return (train_x, train_y), (test_x, test_y)

def _extract_features_from_one_file(fn):
    y, sr = sf.read(fn)
    if sr < FREQUENCY_MAX * 2:
        print("WARNING: sample rate %d is not enough to support max frequency" % (sr, FREQUENCY_MAX))
        return None
    if y.shape[0] < DURATION * sr:
        print("WARNING: %s is less then %d seconds long." % (fn, DURATION))
        return None
    if len(y.shape) > 1:
        y = y[:, 0] # retain only the first channel as if it's mono
    y = y[:DURATION*sr]
    mfcc = _compute_mfcc(y, sr)
    # print("mfcc.shape", mfcc.shape)
    # we use features as average of mfcc over the time of the signal
    mfcc2 = np.mean(mfcc.T, axis=0)
    return mfcc2

# mfcc
# MFCCs are commonly derived as follows:
# 1. Take the Fourier transform of (a windowed excerpt of) a signal.
# 2. Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.
# 3. Take the logs of the powers at each of the mel frequencies.
# 4. Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
# 5. The MFCCs are the amplitudes of the resulting spectrum.

# 1. frequency spectrum:
#    the general range of hearing for human are 20hz to 20k hz. thus, for wave
#    file, you typicall see sample rate of 44K, 22K, 11K etc.
#
# 2. framing:
#    content of the sound signal change over time, thus we do
#    stft over a short time window, typical frame is 25 ms, i.e. 40
#    frames per second, with ~50% overlapping.
#
# 3. number of frequency bins:
#    if a frame contains N_FFT samples, we call it a N_FFT point FFT. The number
#    of frequency bins returned in librosa is 1 + N_FFT/2. The max frequency is
#    Nyquist frequency, which is SR/2. The width of each frequency bin is thus:
#    Nyquist_Frequency/N_FFT_BINS = SR/N_FFT, i.e. number of frames per second
#    as if there's no overlapping
def _compute_mfcc(y, sr):
    #self.print_array_stats(y, "y")
    # num_frequency_bins = 1 + n_fft/2
    # the frequency bins are [0, ..., SR/2]
    # number_of_frames = DURATION * SR / HOP_LENGTH
    #                  ~= DURATION * 4 * SR / N_FFT
    # what's a good N_FFT? a window for about 40 ms, which is about
    # SR / 25 samples
    n_fft = sr // 25
    D = librosa.stft(y, n_fft=n_fft) # Get the STFT matrix
    D = np.abs(D) # The magnitude spectrum
    D = D**2  # The power spectrum
    # The shpe of D is (freq_bins, frames). It has the N_FFT info,
    # but need SR to determine the top frequency. The # of frames
    # just indicate how long the audio is
    #print("PowerSpectrum D.shape", D.shape, "n_fft", n_fft)
    #self.print_array_stats(D, "D")
    # N_FFT will be inferred from the shape of D, the PowerSpectrum
    # sr need to be passed in
    # It seems to make sense to have equal number of mel frequency
    # bins as FFT frequency bins, but the librosa defauts to
    # N_FFT at 2048 and N_MELS at 128, so I just devide by 10
    # should be roughly around 100 to 200
    n_mels = n_fft // 10
    S = librosa.feature.melspectrogram(
        sr=sr,
        S=D,
        n_mels=n_mels,
        fmin=FREQUENCY_MIN,
        fmax=FREQUENCY_MAX)
    #print("S.shape", S.shape, "n_mels", n_mels)
    #self.print_array_stats(S, "S")
    # Now calculate the MFCC coefficients
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(S),
        n_mfcc=N_MFCC)
    return mfcc

if __name__ == "__main__":
    load_features()
