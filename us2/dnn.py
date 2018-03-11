#!/usr/bin/env python

import argparse
import numpy as np
import random
import tensorflow as tf
import time

import urban_sound_data

parser = argparse.ArgumentParser()

n_dim = urban_sound_data.N_MFCC
n_classes = len(urban_sound_data.SOUND_CLASSES)
n_hidden_units_one = 320
n_hidden_units_two = 320
learning_rate = 0.01
num_of_epochs = 200
batch_size = 100

X = tf.placeholder(tf.float32, [None, n_dim], name='X')
Y = tf.placeholder(tf.float32, [None, n_classes], name='Y')

def build_dnn_model():
    sd = 1 / np.sqrt(n_dim)
    W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name='w_1')
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name='b_1')
    h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name='w_2')
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name='b_2')
    h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd), name='w')
    b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name='b')
    logits = tf.matmul(h_2, W) + b
    #y_pred = tf.nn.softmax(tf.matmul(h_2, W) + b)
    y_pred = tf.nn.softmax(logits, name="pred")
    return logits, y_pred

def _shuffle_trainset(train_x, train_y):
    train_x_shuffled, train_y_shuffled = [], []
    idx = list(range(train_x.shape[0]))
    random.shuffle(idx)
    for i in idx:
        train_x_shuffled.append(train_x[i])
        train_y_shuffled.append(train_y[i])
    return np.array(train_x_shuffled), np.array(train_y_shuffled)

def main(argv):
    args = parser.parse_args(argv[1:])
    train_x, train_y = urban_sound_data.load_data([1,2,3,4,5,6,7,8])
    print("train shapes", train_x.shape, train_y.shape)
    test_x, test_y = urban_sound_data.load_data([9, 10])
    print("test shapes", test_x.shape, test_y.shape)

    logits, y_ = build_dnn_model()
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cost_history = np.empty(shape=[1], dtype=float)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    export_dir = '/tmp/v_ear/export-%s' % timestr
    model_dir = export_dir + '/my_model'
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, model_dir, 'vear_model.pbtxt')
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_of_epochs):
            train_x_shuffled, train_y_shuffled = _shuffle_trainset(train_x, train_y)
            for offset in range(0, train_x_shuffled.shape[0], batch_size):
                if offset + batch_size > train_x_shuffled.shape[0]:
                    break
                batch_x = train_x_shuffled[offset:(offset + batch_size), :]
                batch_y = train_y_shuffled[offset:(offset + batch_size), :]
                _, cost = sess.run([optimizer, loss_op], feed_dict={X: batch_x, Y: batch_y})
                cost_history = np.append(cost_history, cost)
            #print("test_x.shape", test_x.shape, "test_y.shape", test_y.shape)
            accuracy_at_epoch = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
            print("done epoch %d, loss=%.3f, accuracy=%.3f" % (epoch, cost_history[-1], accuracy_at_epoch))

        save_path = saver.save(sess, model_dir + "/ckpt")
        print("saved ckpt file %s" % save_path)
        builder.add_meta_graph_and_variables(sess,
                                   [tf.saved_model.tag_constants.TRAINING],
                                   signature_def_map=None,
                                   assets_collection=None)
        # training done, try inference now
        print("done training")

    builder.save()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
