import tensorflow as tf

v = tf.Variable(0, name='my_variable')
with tf.Session() as sess:
    sess = tf.Session()
    tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
