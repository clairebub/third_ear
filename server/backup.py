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

def mfcc_extraction(filepath, order=2, sr=16000, width=None):
    height = 13
    y, sr = librosa.load(filepath, sr=sr)
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=height)
    if width:
        if len(mfcc[0]) > width:
            return None

    mfcc_delta1 = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    if order == 2:
        mfccs = np.concatenate((mfcc, mfcc_delta1, mfcc_delta2))
    if order == 0:
        mfccs = mfcc
    if width:
        mfccs = np.pad(mfccs, ((0, 0), (0, width - len(mfccs[0]))), mode='constant', constant_values=0)
        return mfccs
    else:
        return mfccs, len(mfcc[0])

def command_data_process(rootdir='/home/shane.z/Documents/untitled1/commands/', order=2, sr=16000, width=383):
    data = {}
    print "enter command data process"
    for dirpath, dirname, files in os.walk(rootdir):
        for f in files:
            if f.startswith('.'):
                continue
            if f.startswith('testfile'):
                os.remove(rootdir+'testfile.wav')
                continue
            if f.endswith('.wav'):
                mfccs = mfcc_extraction(dirpath + '/' + f, order, sr, width)
                if mfccs is None:
                    continue
                print f.split('_')
                label_number = int(f.split('_')[0])
                if label_number is None:
                    continue
                if label_number in data:
                    data[label_number].append(mfccs)
                else:
                    data[label_number] = [mfccs]
    return data


def base_data_process(rootdir='/home/shane.z/Documents/untitled/reddots/features/wav.scp', order=2, sr=16000):
    data = {}
    width = 0
    with open(rootdir) as f:
        for line in f:
            label_number = int(line.split()[0].split('_')[-1]) - 31
            if label_number not in range(20):
                continue
            path = line.split()[1]
            mfccs, new_width = mfcc_extraction('/home/shane.z/Documents/untitled/' + path, order, sr)
            if label_number in data:
                data[label_number].append(mfccs)
            else:
                data[label_number] = [mfccs]
            if width < new_width:
                width = new_width
    return data, width


def data_prepare(dic_data, index, width):
    data = []
    labels = []
    for key in index:
        for ele in dic_data[key]:
            mfcc = np.pad(ele,((0, 0), (0, width - len(ele[0]))), mode='constant', constant_values=0)
            labels.append(key)
            data.append(mfcc)
    data = np.array(data)
    labels = np.array(labels)
    data, labels = shuffle(data, labels)
    return data, labels


def base_model_train(train_data, train_labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    for train_index, test_index in sss.split(train_data, train_labels):
        X_train, X_test, y_train, y_test = train_data[train_index], train_data[test_index], train_labels[train_index], train_labels[test_index]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    base_model = cnn_model(X_train, Y_train, X_test, Y_test)
    return base_model


def cnn_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(39, 383, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Max2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    #model.add(Conv2D(64,(3,3),padding='same', input_shape=(39, 383, 1), activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64,(3,3),padding='same', activation='relu'))
    model.add(Max2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Conv2D(128,(3,3),padding='same', activation='relu'))
    model.add(Conv2D(128,(3,3),padding='same', activation='relu'))
    model.add(Max2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Conv2D(256,(3,3),padding='same', activation='relu'))
    model.add(Conv2D(256,(3,3),padding='same', activation='relu'))
    model.add(Max2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # model.add(Conv2D(256,(3,3),padding='same', activation='relu'))
    #model.add(Conv2D(512,(3,3),padding='same', activation='relu'))
    #model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(1024,(3,3),padding='same', activation='relu'))
    #model.add(Max2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    adam=keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print (model.summary())
    filepath = "weights.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list=[checkpoint]
    model.fit(X_train, Y_train, batch_size=32, epochs=30, callbacks=callbacks_list,
              verbose=1, validation_split=0.2)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print ' '
    print 'Test score: ', score[0]
    print 'Test accuracy: ', score[1]

    bestmodel = load_model('weights.best.hdf5')
    bestscore = bestmodel.evaluate(X_test, Y_test, verbose=1)
    print ' '
    print 'Test score of Best Model: ', bestscore[0]
    print 'Test accuracy of Best Model: ', bestscore[1]
    return model

def data_fine_tuning(dic_data, index, size=5, mode='other'):
    data = []
    labels = []
    for key in index:
        for ele in dic_data[key]:
            mfcc = np.pad(ele, ((0, 0), (0, width - len(ele[0]))), mode='constant', constant_values=0)
            labels.append(key)
            data.append(mfcc)
    data = np.array(data)
    labels = np.array(labels)
    train_size = len(index) * size
    if mode == 'command':
        data, tun_labels_train = shuffle(data, labels)
        tun_data_train = data.reshape(data.shape[0], data.shape[1],
                                      data.shape[2], 1)
    else:
        # if train_size < data.shape[0]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        for train_index, test_index in sss.split(data, labels):
            tun_data_train, tun_labels_train = data[train_index], labels[train_index]
        tun_data_train = tun_data_train.reshape(tun_data_train.shape[0], tun_data_train.shape[1], tun_data_train.shape[2], 1)
            #tun_labels_train = np_utils.to_categorical(tun_labels_train, num_classes)

        # else:
        #     data, tun_labels_train = shuffle(data, labels)
        #     tun_data_train = data.reshape(data.shape[0], data.shape[1],
        #                                             data.shape[2], 1)
    return tun_data_train, tun_labels_train


def get_new_data(new_files, dir='/home/shane.z/Documents/untitled1/commands/', width=383):
    command_data = {}
    for file in new_files:
        label_number = file.split('_')[0]
        mfccs = mfcc_extraction(dir + file, width=width)

        if label_number in data:
            command_data[label_number].append(mfccs)
        else:
            command_data[label_number] = [mfccs]
    return command_data


def prepare_test_data(rootdir='/home/shane.z/Documents/untitled1/commands/testfile.wav', width=383):
    mfccs = mfcc_extraction(rootdir, width=width)
    mfcc = np.pad(mfccs, ((0, 0), (0, width - len(mfccs[0]))), mode='constant', constant_values=0)
    test_data = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
    return test_data


def fine_tuning_training(model, train_data, train_labels, batch_size=32, epochs=250):
    with tf.device('/cpu:0'):
        model.pop()
        model.add(Dense(num_classes, activation='softmax', name='new_softmax'))
        tuning_filepath = "tuning_weights.best.hdf5"
        tuning_checkpoint = keras.callbacks.ModelCheckpoint(tuning_filepath, monitor='acc', verbose=1,
                                                            save_best_only=True, mode='max')
        tuning_callbacks_list = [tuning_checkpoint]
        adam = keras.optimizers.Adam(lr=0.001)
        print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, callbacks=tuning_callbacks_list,
                      verbose=1)

def check_string(string):
    result = None
    with open("label_record.txt") as f:
        found = False
        for line in f:  #iterate over the file one line at a time(memory efficient)
            #if re.search("\b{0}\b".format(string), line):    #if string found is in current line then print it
             if string in line:
                # print line.split()[1]
                result = line
                found = True
                break
        if not found:
            print('The command cannot be found!')
        return result

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # K.set_session(sess)

    if False:
        action_host = '192.168.1.21'
        action_port = 50008
        action_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        action_s.settimeout(None)
        try:
            action_s.connect((action_host, action_port))
            print "Connected to action host"
        except Exception as e:
            print e
            sys.exit()
            print('Unable to connect action host')


    # train the base model
    if True:
        print str(datetime.now())
        data, width = base_data_process()
        print "base data labels and width: ", len(data.keys()), width

        width = 383
        org = range(10)
        train_data, train_labels = data_prepare(data, org, width)
        print train_data.shape, train_labels
        base_model = base_model_train(train_data, train_labels)
        sys.exit(0)

    base_model = load_model('weights.best.hdf5')
    base_tun_data_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # tun_data_base, tun_labels_base = data_fine_tuning(data, base_tun_data_index, 5)
    # tun_data, tun_labels = tun_data_base[:], tun_labels_base[:]
    print "Base model prepared"

    HOST = ''
    SOCKET_LIST = []
    RECV_BUFFER = 4096
    PORT = 5021
    VERBOSE = False
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(10)
    SOCKET_LIST.append(server_socket)
    print "Ready to connect"



    exist_files = set()
    while 1:
        # get the list sockets which are ready to be read through select
        ready_to_read, ready_to_write, in_error = select.select(SOCKET_LIST, [], [], 0)
        for sock in ready_to_read:
            # a new connection request received
            if sock == server_socket:
                sockfd, addr = server_socket.accept()
                SOCKET_LIST.append(sockfd)
                print("Client (%s, %s) connected" % addr)
            # a message from a client, not a new connection
            else:
                # process data recieved from client,
                try:
                    # receiving data from the socket.
                    msg = sock.recv(RECV_BUFFER)
                    if msg == 'train':
                        print msg
                        print str(datetime.now())
                        # if this is the first time to add commands
                        if not exist_files:
                            print "no exist command"
                            new_command_data = command_data_process(width=width)
                            exist_files = set(
                                os.listdir("/home/shane.z/Documents/untitled1/commands/"))
                            print "command data labels: ", new_command_data.keys()

                        else:
                            new_files = set(os.listdir("/home/shane.z/Documents/untitled1/commands/")) - exist_files
                            new_command_data = get_new_data(new_files)
                            exist_files = set(os.listdir("/home/shane.z/Documents/untitled1/commands/"))

                            print "new command data labels: ", new_command_data.keys()
                        # print sorted(exist_files)[-1]
                        num_classes = int(sorted(exist_files)[-1].split('_')[0]) + 1
                        print num_classes
                        new_command_list = new_command_data.keys()
                        tun_data_commands, tun_labels_commands = data_fine_tuning(new_command_data, new_command_list, mode='command')

                        print "commands data complete"
                        #tun_data = np.concatenate((tun_data, tun_data_commands))
                        #tun_labels = np.concatenate((tun_labels, tun_labels_commands))
                        tun_data = tun_data_commands
                        tun_labels = tun_labels_commands
                        print "tun_data, tun_labels complete"
                        final_tun_labels = np_utils.to_categorical(tun_labels, num_classes)

                        print "Begin to train"
                        print str(datetime.now())
                        tun_model = load_model('/home/shane.z/Documents/untitled/weights.best.hdf5')
                        for i in tun_model.layers:
                            if i.name == 'dense_2':
                                break
                            i.trainable = False
                        print tun_model.summary()
                        fine_tuning_training(tun_model, tun_data, final_tun_labels, batch_size=5, epochs=100)

                        print "training done, you can test now"
                        done_msg = "train ok"
                        sock.send(done_msg)

                    if msg == "test prepared":
                        print msg
                        try:
                            test_model = load_model('/home/shane.z/Documents/untitled/tuning_weights.best.hdf5')
                        except:
                            test_model = load_model('/home/shane.z/Documents/untitled/weights.best.hdf5')
                        print "begin", str(datetime.now())
                        test_data = prepare_test_data()
                        print "begin predic", str(datetime.now())
                        with tf.device('/cpu:0'):
                            prediction = test_model.predict(test_data)
                            class_number = np.argmax(prediction, axis=1)
                            print str(datetime.now())
                            print class_number
                            sock.send("test done")
                            sock.send(str(class_number[0]))
                            print "send"

                        command_number = int(class_number[0])
                        command_try = {}
                        command_try['request_id'] = '2209'
                        command_try['action_id'] = 'qeitsxrhho'
                        command_try['timestamp'] = 1501281162
                        command_try['intent'] = {}
                        command_try['intent']['name'] = command_number / 100
                        command_try['intent']['slots'] = {}
                        command_try['intent']['slots']['type'] = command_number % 1000
                        command_try['intent']['slots']['params'] = []
                        if command_number / 100 == 201:
                            command_try['intent']['slots']['params'] = [7]
                        if command_number / 100 == 203:
                            command_try['intent']['slots']['params'] = [5]
                        if command_number / 100 == 204:
                            command_try['intent']['slots']['params'] = ['maps']
                        if command_number / 100 == 205:
                            command_try['intent']['slots']['params'] = ['4089094998',
                                                                        "I'm busy now, call you later"]
                        if command_number / 100 == 206:
                            command_try['intent']['slots']['params'] = ['www.samsung.com']
                        if command_number / 100 == 207:
                            command_try['intent']['slots']['params'] = [1]

                        data = json.dumps(command_try)
                        if False: # no mobile backend to talk to
                            try:
                                action_s.send(data)
                                print "mobile sent"
                            except Exception as e:
                                print e

                        #while 1:
                        try:
                            change_test_name = sock.recv(RECV_BUFFER)
                            if change_test_name == 'change name':
                                print change_test_name
                                new_name = sock.recv(RECV_BUFFER)
                                # if int(new_name.split('_')[0]) < 10:
                                #     os.rename('reddots/commands/testfile.wav', 'reddots/org_commands/' + new_name + '.wav')
                                os.rename('/home/shane.z/Documents/untitled1/commands/testfile.wav',
                                          '/home/shane.z/Documents/untitled1/commands/' + new_name + '.wav')
                                break
                        except KeyboardInterrupt:
                            break

                except Exception as e:
                    print e
                    continue
    # server_socket.close()
