server.py:

mfcc_extraction: #use to extract mfcc feature
"""
filepath: file path of wav
order: order of mfcc feature to extract, default=2
sr: sample rate, default 16k
width: mfcc feature’s dimension, default=None
"""

command_data_process: # use to process new command data to get label and utilize mfcc_extraction to get mfcc feature.
"""
rootdir: direction of new command data
order: order of mfcc feature to extract, default=2
sr: sample rate, default 16k
width: mfcc feature’s dimension, default=383 which is the biggest of base data
"""

base_data_process: # use to process base command data to get label and utilize mfcc_extraction to get mfcc feature.
"""
rootdir: direction of base command data
order: order of mfcc feature to extract, default=2
sr: sample rate, default 16k
"""

data_prepare: #process data into format of model input
"""
dic_data: command data in dictionary, with key is the label,value is the data
index: all voice command labels
width: the dimension we set to voice data
"""

base_model_train: # to train a base a model
"""
train_data: training data
train_labels: labels of training data
"""

cnn_model: # model used to train
"""
X_train: training data
Y_train: labels of training data
X_test: validation data
Y_test: labels of validation data
"""

cnn_model: # model used to train
"""
X_train: training data
Y_train: labels of training data
X_test: validation data
Y_test: labels of validation data
"""

data_fine_tuning: # use to re-train model with new command data
cnn_model: # model used to train
"""
dic_data: training data in dictionary format, key is label
index: labels that you want to add to tuning data
size: how much you want to add for each kind of data
mode: [‘other’, ‘command’], ‘other’ means data is from base data, command means from new data
"""

get_new_data: #use to precess the latest new command data without other new new data
"""
new_files: list of new commands name
dir: directory of these new commands
width: column dimension we want new command data to be(the length of longest voice)
"""

prepare_test_data: # use to process test data file
"""
rootdir: file path
width: default 383, same to training data
"""

fine_tuning_training # retrain model with new data
"""
mode: original model
train_data: new training data
train_labels: new training labels
batch_size: batch_size
epochs: epochs
"""

check_string: #use to check if the new command exists
"""
string: command you want to check
"""