import tensorflow as tf

with tf.Session() as sess:
  saver = tf.train.import_meta_graph('/tmp/tf_training/ckpt/rjmodel.ckpt.meta')
  saver.restore(sess, tf.train.latest_checkpoint('/tmp/tf_training/ckpt/'))
  print(sess.run('W:0'))
