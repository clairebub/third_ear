#!/usr/bin/env python

import glob
import librosa
import numpy as np
import os
import pickle
import random
import soundfile as sf
import sys
import tensorflow as tf
import time

from collections import defaultdict
from time import gmtime, strftime

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

# float hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 1024, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_of_epochs", 30, "Number of epochs during training.")
tf.app.flags.DEFINE_integer("n_hidden_units_one", 320, "Hidden layer one size.")
tf.app.flags.DEFINE_integer("n_hidden_units_two", 320, "Hidden layer two size.")
tf.app.flags.DEFINE_boolean("training", True, "Set to True for training and False for inference")

FLAGS = tf.app.flags.FLAGS

# we will take 3 seconds audio and look at melspectrogram at some mel frequencies
# in the range of (FREQUENCY_MIN, FREQUENCY_MAX)
DURATION = 3
N_MFCC= 40
FREQUENCY_MIN = 20
FREQUENCY_MAX = 11000

class DNNModeling(object):

    def __init__(self):
        self.classes = {
            0 : 'air_conditioner',
            1 : 'car_horn',
            2 : 'children_playing',
            3 : 'dog_bark',
            4 : 'drilling',
            5 : 'engine_idling',
            6 : 'gun_shot',
            7 : 'jackhammer',
            8 : 'siren',
            9 : 'street_music'
        }

    def load_graph(self, frozen_graph_filename="/tmp/stem/export-current/sound_model.pb"):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="train")
        return graph

    def build_dnn_model(self):
        sd = 1 / np.sqrt(self.n_dim)

        W_1 = tf.Variable(tf.random_normal([self.n_dim, FLAGS.n_hidden_units_one], mean=0, stddev=sd), name='w_1')
        b_1 = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_one], mean=0, stddev=sd), name='b_1')
        h_1 = tf.nn.tanh(tf.matmul(self.X, W_1) + b_1)

        W_2 = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_one, FLAGS.n_hidden_units_two], mean=0, stddev=sd), name='w_2')
        b_2 = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_two], mean=0, stddev=sd), name='b_2')
        h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

        W = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_two, len(self.classes)], mean=0, stddev=sd), name='w')
        b = tf.Variable(tf.random_normal([len(self.classes)], mean=0, stddev=sd), name='b')
        logits = tf.matmul(h_2, W) + b
        y_pred = tf.nn.softmax(tf.matmul(h_2, W) + b)
        return logits, y_pred

    def compute_mfcc(self, y, sr):
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

    def extract_features_from_one_file(self, fn):
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
        mfcc = self.compute_mfcc(y, sr)
        # print("mfcc.shape", mfcc.shape)
        # we use features as average of mfcc over the time of the signal
        mfcc2 = np.mean(mfcc.T, axis=0)
        return mfcc2

    def extract_features(self):
        data_dir = "../data"
        label_dirs = ['1', '2']
        features, labels = np.zeros(0), np.zeros(0, dtype=int)
        for label_dir in label_dirs:
            for fn in glob.glob(os.path.join(data_dir, label_dir, "*.wav")):
                mfcc2 = self.extract_features_from_one_file(fn)
                if mfcc2 is None:
                    continue
                features = np.append(features, mfcc2)
                labels= np.append(labels, int(label_dir))
                # print("mfcc2.shape", mfcc2.shape)
                # print(mfcc2)
        features = features.reshape(-1, N_MFCC)
        labels = self.one_hot_encode(labels)
        print("features.shape", features.shape)
        print("labels.shape", labels.shape)
        print("labels", labels)
        return features, labels

    def train(self):
        #features = pickle.load(open("%s/audio/dnn_features.p" % (os.getcwd()), "rb"))
        #labels = pickle.load(open("%s/audio/dnn_labels.p" % (os.getcwd()), "rb"))
        features, labels = self.extract_features()
        self.data = {'features': features, 'labels': labels}
        self.n_dim = features.shape[1]
        self.X = tf.placeholder(tf.float32, [None, self.n_dim], name='X')
        self.Y = tf.placeholder(tf.float32, [None, len(self.classes)], name='Y')

        # finish the graph model and keep track of some stats we are interested
        logits, y_ = self.build_dnn_model()
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        cost_history = np.empty(shape=[1], dtype=float)

        # split the train/test data sets
        rnd_indices = np.random.rand(len(self.data['labels'])) < 0.70
        train_x = self.data['features'][rnd_indices]
        train_y = self.data['labels'][rnd_indices]
        test_x = self.data['features'][~rnd_indices]
        test_y = self.data['labels'][~rnd_indices]

        timestr = time.strftime("%Y%m%d-%H%M%S")
        export_dir = '/tmp/stem/export-%s' % timestr
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
#        saver = tf.train.Saver()
        with tf.Session() as sess:
#            tf.train.write_graph(sess.graph_def, model_dir, 'model.pbtxt')
            sess.run(tf.global_variables_initializer())
            for epoch in range(FLAGS.num_of_epochs):
                train_x_shuffled, train_y_shuffled = self._shuffle_trainset(train_x, train_y)
                for offset in range(0, train_x_shuffled.shape[0], FLAGS.batch_size):
                    if offset + FLAGS.batch_size > train_x_shuffled.shape[0]:
                        break
                    batch_x = train_x_shuffled[offset:(offset + FLAGS.batch_size), :]
                    batch_y = train_y_shuffled[offset:(offset + FLAGS.batch_size), :]
                    _, cost = sess.run([optimizer, loss_op], feed_dict={self.X: batch_x, self.Y: batch_y})
                    cost_history = np.append(cost_history, cost)
                print("test_x.shape", test_x.shape, "test_y.shape", test_y.shape)
                accuracy_at_epoch = sess.run(accuracy, feed_dict={self.X: test_x, self.Y: test_y})
                print("done epoch %d, loss=%.3f, accuracy=%.3f" % (epoch, cost_history[-1], accuracy_at_epoch))

#            save_path = saver.save(sess, model_dir + "/ckpt")
#            print("saved ckpt file %s" % save_path)
            builder.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.TRAINING],
                                       signature_def_map=None,
                                       assets_collection=None)
            # training done, try inference now
            print("done training")

        builder.save()

    def _shuffle_trainset(self, train_x, train_y):
        train_x_shuffled, train_y_shuffled = [], []
        idx = list(range(train_x.shape[0]))
        random.shuffle(idx)
        for i in idx:
            train_x_shuffled.append(train_x[i])
            train_y_shuffled.append(train_y[i])
        return np.array(train_x_shuffled), np.array(train_y_shuffled)

    def print_array_stats(self, a, name="a"):
        print(name, a)
        for p in range(10, 100, 10):
            print("%s.p(%d)=%f" % (name, p, np.percentile(a, p)))

    def one_hot_encode(self, labels):
        print("one_hot", labels.shape, labels)
        n_labels = len(labels)
        n_classes = len(self.classes)
        if np.max(labels) >= n_classes:
            raise ValueError('label.max=%d, greater than n_classes=%d'%(np.max(labels), n_classes))
        one_hot_encode = np.zeros((n_labels, n_classes))
        one_hot_encode[np.arange(n_labels), labels] = 1

        return one_hot_encode

    def inference(self):
        graph = self.load_graph()
        node_names = [n.name for n in graph.as_graph_def().node]
        print("graph nodes:", node_names)
        x = graph.get_tensor_by_name('train/X:0')
        y = graph.get_tensor_by_name('train/Y:0')
        print("x", x)
        print("y", y)

        test_file = "../data/1/baby-crying-01.wav"
        mfcc2 = self.extract_features_from_one_file(test_file)
        if mfcc2 is None:
            print("Error: failed to extract mfcc features from file:", test_file)
            return
        y_value =  np.random.uniform(0, 1, 10)
        print("y_value", y_value)

        # We launch a Session
        with tf.Session(graph=graph) as sess:
            # Note: we don't nee to initialize/restore anything
            # There is no Variables in this graph, only hardcoded constants
            y_out = sess.run(y, feed_dict={
                x: [mfcc2], # < 45
                y: [y_value]
            })
            # I taught a neural net to recognise when a sum of numbers is bigger than 45
            # it should return False in this case
            print(y_out) # [[ False ]] Yay, it works!
            y_ind = np.argmax(y_out)
            pred = self.classes[y_ind]
            print("***** inferece result is: ", pred)

def main():
    dnn = DNNModeling()

    if FLAGS.training is True:
        dnn.train()
    else:
        dnn.inference()

if __name__ == '__main__':
    main()
