import glob
import os
import librosa
import pickle
import numpy as np

SAMPLE_RATE = 2000
DURATION = 1
N_FFT = 200

def extract_feature(file_name):
    print("extract_feature: [%s]" % file_name)
    y, sr = librosa.load(
        file_name,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION)
    print("y.shape=[%s], sample_rate=[%d]" % (y.shape, sr))
    #stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=N_FFT//4))
    #print("stft.shape=", stft.shape)
    #print("stft", stft)
    mfccs = np.mean(librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=40).T,
        axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    return mfccs, chroma, mel, contrast


def parse_audio_files(data_dir, sub_dirs, file_ext='*.wav'):
    # features, labels = np.empty((0, 193)), np.empty(0)
    features, labels = np.zeros((0, 187)), np.empty(0)
    print("data_dir: [%s]" % data_dir)
    print("sub_dirs: [%s]" % sub_dirs)

    for label in sub_dirs:
        labeled_dir = os.path.join(data_dir, label)
        print("Processing label [%s] in [%s]..." % (label, labeled_dir))
        for fn in glob.glob(os.path.join(labeled_dir, file_ext)):
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
    print("hello")
    features, labels = data_preparation('../data', ['1', '2'])
    # save feature and label files
    #pickle.dump(features, open("../UrbanSound8K/audio/dnn_features.p", "wb"))
    #pickle.dump(labels, open("../UrbanSound8K/audio/dnn_labels.p", "wb"))

    # print("Print features:")
    # print(features)
    # print("\nPrint labels:")
    # print(labels)


if __name__ == '__main__':
    main()
