import tensorflow as tf

with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], "/tmp/SavedModel")
  print(sess.run('W:0'))
