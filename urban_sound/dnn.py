import os
import sys
import pickle
import random
import tensorflow as tf
import numpy as np
from collections import defaultdict
from time import gmtime, strftime

from dnn_data_processing import extract_feature

# python sample w/o replacement
# x = range(10)
# x2 = ramdonm.sample(x, 3)
# in-place random shuffle of a list
# x = list(range(10))
# random.shuffle(x)
# split the data set into train and test
# x = np.random.rand(100) < 0.7
# train_set = data[x]
# test_set = data[~x]

# float hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 1024, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_of_epochs", 30, "Number of epochs during training.")
tf.app.flags.DEFINE_integer("n_hidden_units_one", 320, "Hidden layer one size.")
tf.app.flags.DEFINE_integer("n_hidden_units_two", 320, "Hidden layer two size.")
# string hyperparameters
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoint", "Checkpoint directory.")

FLAGS = tf.app.flags.FLAGS

class DNNModeling(object):
    def __init__(self):
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

    def train(self):
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

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.train.write_graph(sess.graph_def, '/tmp/urban-sound', 'model.pbtxt')
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
                save_path = saver.save(sess, "/tmp/urban_sound_ckpt", global_step=epoch)
                print("saved ckpt file %s" % save_path)
            print("done training")

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
