import glob
import librosa
import numpy as np
import os
import pickle
import random
import sys
import tensorflow as tf

from collections import defaultdict
from time import gmtime, strftime

# mfcc
#
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
# string hyperparameters
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoint", "Checkpoint directory.")

FLAGS = tf.app.flags.FLAGS

DURATION = 3
SAMPLE_RATE = 22050
N_FFT = 512 # roughly N_FFT // SAMPLE_RATE ~= 0.025 sec ~= 25 ms
N_MFCC= 40

class DNNModeling(object):
    def __init__(self):
        #features = np.zeros((0, 10)) # pickle.load(open("%s/audio/dnn_features.p" % (os.getcwd()), "rb"))
        #labels = np.zeros((0, 10)) #pickle.load(open("%s/audio/dnn_labels.p" % (os.getcwd()), "rb"))
        features = pickle.load(open("%s/audio/dnn_features.p" % (os.getcwd()), "rb"))
        labels = pickle.load(open("%s/audio/dnn_labels.p" % (os.getcwd()), "rb"))

        self.data = {'features': features, 'labels': labels}
        self.n_dim = features.shape[1]
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
        self.X = tf.placeholder(tf.float32, [None, self.n_dim])
        self.Y = tf.placeholder(tf.float32, [None, len(self.classes)])

    def build_dnn_model(self):
        sd = 1 / np.sqrt(self.n_dim)

        W_1 = tf.Variable(tf.random_normal([self.n_dim, FLAGS.n_hidden_units_one], mean=0, stddev=sd))
        b_1 = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_one], mean=0, stddev=sd))
        h_1 = tf.nn.tanh(tf.matmul(self.X, W_1) + b_1)

        W_2 = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_one, FLAGS.n_hidden_units_two], mean=0, stddev=sd))
        b_2 = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_two], mean=0, stddev=sd))
        h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

        W = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_two, len(self.classes)], mean=0, stddev=sd))
        b = tf.Variable(tf.random_normal([len(self.classes)], mean=0, stddev=sd))
        logits = tf.matmul(h_2, W) + b
        y_pred = tf.nn.softmax(tf.matmul(h_2, W) + b)
        return logits, y_pred

    def extract_features(self):
        data_dir = "../data"
        label_dirs = ['1', '2']

        for label_dir in label_dirs:
            for fn in glob.glob(os.path.join(data_dir, label_dir, "*.wav")):
                y, sr = librosa.load(
                    fn,
                    sr=SAMPLE_RATE,
                    mono=True,
                    duration=DURATION)
                print("y.shape=[%s], sample_rate=[%d], fn=[%s]" % (y.shape, sr, fn))
                stft = np.abs(librosa.stft(y, n_fft=N_FFT))
                # its shape is (num_frequency_bins, num_of_frames), where
                # num_frequency_bins = 1 + n_fft/2
                # number_of_frames = DURATION * SR / HOP_LENGTH
                #                  ~= DURATION * 4 * SR / N_FFT
                #                  ~= DURATION * 4 * 40
                print("sftf.shape", stft.shape)
                mfccs = librosa.feature.mfcc(
                    y=y,
                    sr=SAMPLE_RATE,
                    n_fft=N_FFT,
                    n_mfcc=N_MFCC)
                # its shape (n_mfcc, number_of_frames)
                print("mfccs.shape", mfccs.shape)
                # mel-scaled spectrogram
                mel = librosa.feature.melspectrogram(
                    y=y,
                    sr=SAMPLE_RATE,
                    n_fft=N_FFT,
                    hop_length=N_FFT//4,
                    power=2.0, **kwargs)[source]
                # filter bank matrix
                #librosa.filters.mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1)
                # return M : np.ndarray [shape=(n_mels, 1 + n_fft/2)]

    def train(self):
        # extract the features
        #self.extract_features()
        #if True:
        #    sys.exit(0)
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

        # saver = tf.train.Saver()
        builder = tf.saved_model.builder.SavedModelBuilder('/tmp/stem/model')
        with tf.Session() as sess:
            tf.train.write_graph(sess.graph_def, '/tmp/stem/model', 'model.pbtxt')
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
                accuracy_at_epoch = sess.run(accuracy, feed_dict={self.X: test_x, self.Y: test_y})
                print("done epoch %d, loss=%.3f, accuracy=%.3f" % (epoch, cost_history[-1], accuracy_at_epoch))
                #save_path = saver.save(sess, "/tmp/urban_sound_ckpt", global_step=epoch)
                #print("saved ckpt file %s" % save_path)
            print("done training")
            builder.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.TRAINING],
                                       signature_def_map=None,
                                       assets_collection=None)
    def inference(self):
        # load the saved model
        # extract the features

        pass

    def _shuffle_trainset(self, train_x, train_y):
        train_x_shuffled, train_y_shuffled = [], []
        idx = list(range(train_x.shape[0]))
        random.shuffle(idx)
        for i in idx:
            train_x_shuffled.append(train_x[i])
            train_y_shuffled.append(train_y[i])
        return np.array(train_x_shuffled), np.array(train_y_shuffled)

def main():
    dnn = DNNModeling()
    dnn.train()

if __name__ == '__main__':
    main()
