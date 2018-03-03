import os
import sys
import pickle
import random
import tensorflow as tf
import numpy as np
from collections import defaultdict
from time import gmtime, strftime

from dnn_data_processing import extract_feature

try:
    # import pynvml in anaconda
    import importlib.machinery

    anaconda3_path = os.path.abspath(sys.executable + "/../../")
    pynvml_path = anaconda3_path + '/lib/python3.5/site-packages/'
    sys.path.append(pynvml_path)

    loader = importlib.machinery.SourceFileLoader('my_pynvml', pynvml_path + 'pynvml.py')
    my_pynvml = loader.load_module()

    import my_pynvml
except:
    pass

# float hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
# tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.98, "Learning rate decays by this much.")
# tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
# tf.app.flags.DEFINE_float("dropout", 0.2, "dropout rate.")
# int hyperparameters
tf.app.flags.DEFINE_integer("batch_size", 1024, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_of_epochs", 50000, "Number of epochs during training.")
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
tf.app.flags.DEFINE_boolean("training", True, "Set to True for training and False for inference")

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

    def get_default_gpus(self):
        my_pynvml.nvmlInit()
        deviceCount = my_pynvml.nvmlDeviceGetCount()

        gpu_usage = defaultdict(int)
        for i in range(deviceCount):
            handle = my_pynvml.nvmlDeviceGetHandleByIndex(i)
            info = my_pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpu_usage[i] = info.used

        sorted_gpu_usage = sorted(gpu_usage, key=gpu_usage.get)

        return sorted_gpu_usage[0]

    def shuffle_trainset(self, train_x, train_y):
        train_x_shuf = []
        train_y_shuf = []
        index = range(train_x.shape[0])
        index_shuf = random.sample(list(index), len(index))
        for i in index_shuf:
            train_x_shuf.append(train_x[i])
            train_y_shuf.append(train_y[i])

        return np.array(train_x_shuf), np.array(train_y_shuf)

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
        # find an available gpu
        #gpu = self.get_default_gpus()
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        rnd_indices = np.random.rand(len(self.data['labels'])) < 0.70

        train_x = self.data['features'][rnd_indices]
        train_y = self.data['labels'][rnd_indices]
        test_x = self.data['features'][~rnd_indices]
        test_y = self.data['labels'][~rnd_indices]

        now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        # pickle.dump(train_x, open("../UrbanSound8K/audio/%s_train_dnn_features.p" % now, "wb"))
        # pickle.dump(train_y, open("../UrbanSound8K/audio/%s_train_dnn_labels.p" % now, "wb"))
        # pickle.dump(test_x, open("../UrbanSound8K/audio/%s_test_dnn_features.p" % now, "wb"))
        # pickle.dump(test_y, open("../UrbanSound8K/audio/%s_test_dnn_labels.p" % now, "wb"))

        print("Training set size: %d" % train_x.shape[0])
        print("Testing set size: %d" % test_x.shape[0])

        training_iterations = FLAGS.num_of_epochs * train_x.shape[0]

        # build model
        logits, y_ = None, None
        if FLAGS.model == 'dnn':
            logits, y_ = self.build_dnn_model()

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)

        saver = tf.train.Saver()

        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        cost_history = np.empty(shape=[1], dtype=float)

        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        with tf.Session(config=config_) as sess:
            print("Initializing parameters...")
            sess.run(tf.global_variables_initializer())

            train_x_shuf, train_y_shuf = train_x, train_y
            for itr in range(training_iterations):

                offset = (itr * FLAGS.batch_size) % train_y.shape[0]

                # start a new batch
                if offset < FLAGS.batch_size:
                    offset = 0
                    train_x_shuf, train_y_shuf = self.shuffle_trainset(train_x, train_y)

                if offset == 0:
                    print('Epoch %d: ' % int((itr * FLAGS.batch_size) / train_y_shuf.shape[0]), end='')

                batch_x = train_x_shuf[offset:(offset + FLAGS.batch_size), :]
                batch_y = train_y_shuf[offset:(offset + FLAGS.batch_size), :]

                _, cost = sess.run([optimizer, loss_op], feed_dict={self.X: batch_x, self.Y: batch_y})
                cost_history = np.append(cost_history, cost)

                # print result at the end of each epoch
                if offset + FLAGS.batch_size >= train_y_shuf.shape[0]:
                    print('Training loss: ', cost, '\tTest accuracy: ', sess.run(accuracy, feed_dict={self.X: test_x, self.Y: test_y}))

                    saver.save(sess, "%s/checkpoint/%s_%s_model.ckpt" % (os.getcwd(), now, FLAGS.model))

    def inference(self):
        if FLAGS.use_gpu is True:
            # find an available gpu
            gpu = self.get_default_gpus()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # build model
        logits, y_ = None, None
        if FLAGS.model == 'dnn':
            logits, y_ = self.build_dnn_model()

        pred_class = tf.argmax(y_, 1)

        if FLAGS.use_gpu is True:
            config_ = tf.ConfigProto()
        else:
            config_ = tf.ConfigProto(device_count={'GPU': 0})
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        with tf.Session(config=config_) as sess:
            # load model
            saver = tf.train.Saver()
            saver.restore(sess, "%s/%s/%s" % (os.getcwd(), FLAGS.checkpoint_dir, FLAGS.trained_model))
            print("%s model loaded." % FLAGS.trained_model)

            while True:
                test_data_file = input('Input test file name: ')

                if test_data_file == 'quit':
                    break
                else:
                    test_data = '%s/%s' % (FLAGS.data_dir, test_data_file)

                    # extract features from test_data
                    mfccs, chroma, mel, contrast = extract_feature(test_data)
                    feature = np.hstack([mfccs, chroma, mel, contrast])
                    try:
                        label = test_data.split('-')[1]
                    except:
                        label = -1

                    if label == -1:
                        print('Predicted class: ', self.classes[sess.run(pred_class, feed_dict={self.X: [feature]})[0]])
                    else:
                        print('Predicted class: ', self.classes[sess.run(pred_class, feed_dict={self.X: [feature]})[0]], '\t True class: ', self.classes[label])


def main():
    sound_classifier = DNNModeling()

    if FLAGS.training is True:
        sound_classifier.train()
    else:
        sound_classifier.inference()


if __name__ == '__main__':
    main()
