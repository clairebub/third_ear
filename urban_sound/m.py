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

import os
import sys
import pickle
import random
import tensorflow as tf
import numpy as np
from collections import defaultdict
from time import gmtime, strftime

from dnn_data_processing import extract_feature

# float hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 1024, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_of_epochs", 50, "Number of epochs during training.")
tf.app.flags.DEFINE_integer("n_hidden_units_one", 320, "Hidden layer one size.")
tf.app.flags.DEFINE_integer("n_hidden_units_two", 320, "Hidden layer two size.")
# string hyperparameters
tf.app.flags.DEFINE_string("model", "dnn", "Model selection.")
tf.app.flags.DEFINE_string("data_dir", "audio", "Training data directory.")
tf.app.flags.DEFINE_string("feature_type", "dnn", "Feature type of sound data.")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoint", "Checkpoint directory.")
tf.app.flags.DEFINE_string("trained_model", "2017-12-13-03-02-11_dnn_model.ckpt", "Trained model.")
# # boolean hyperparameters
tf.app.flags.DEFINE_boolean("use_gpu", False, "Set to True for using GPU to do inference and False for using CPU")
tf.app.flags.DEFINE_boolean("training", False, "Set to True for training and False for inference")

FLAGS = tf.app.flags.FLAGS


class DNNModeling(object):

    def __init__(self):
        features = pickle.load(open("%s/%s/%s_features.p" % (os.getcwd(), FLAGS.data_dir, FLAGS.feature_type), "rb"))
        labels = pickle.load(open("%s/%s/%s_labels.p" % (os.getcwd(), FLAGS.data_dir, FLAGS.feature_type), "rb"))

        self.data = {'features': features, 'labels': labels}
        self.n_dim = features.shape[1]
        self.n_classes = 10

        self.X = tf.placeholder(tf.float32, [None, self.n_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.n_classes])

        self.classes = defaultdict(str)
        self.class_name()
        print("features.shape", features.shape)
        print("labels.shape", labels.shape)

    def class_name(self):
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

    def build_dnn_model(self):
        sd = 1 / np.sqrt(self.n_dim)

        W_1 = tf.Variable(tf.random_normal([self.n_dim, FLAGS.n_hidden_units_one], mean=0, stddev=sd))
        b_1 = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_one], mean=0, stddev=sd))
        h_1 = tf.nn.tanh(tf.matmul(self.X, W_1) + b_1)

        W_2 = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_one, FLAGS.n_hidden_units_two], mean=0, stddev=sd))
        b_2 = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_two], mean=0, stddev=sd))
        h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

        W = tf.Variable(tf.random_normal([FLAGS.n_hidden_units_two, self.n_classes], mean=0, stddev=sd))
        b = tf.Variable(tf.random_normal([self.n_classes], mean=0, stddev=sd))
        logits = tf.matmul(h_2, W) + b
        y_pred = tf.nn.softmax(tf.matmul(h_2, W) + b)

        return logits, y_pred

    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, "%s/c2/%s_%s_model.ckpt" % (os.getcwd(), now, FLAGS.model))
            print("done training")

def main():
    sound_classifier = DNNModeling()
    sound_classifier.train()

if __name__ == '__main__':
    main()
