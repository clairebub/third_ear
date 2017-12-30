import tensorflow as tf

I = tf.placeholder(tf.float32, shape=[None,3], name='I') # input
W = tf.Variable(tf.zeros(shape=[3,2]), dtype=tf.float32, name='W') # weights
b = tf.Variable(tf.zeros(shape=[2]), dtype=tf.float32, name='b') # biases
O = tf.nn.relu(tf.matmul(I, W) + b, name='O') # activation / output
init_op = tf.global_variables_initializer()

builder = tf.saved_model.builder.SavedModelBuilder('/tmp/SavedModel/')

with tf.Session() as sess:
  sess.run(init_op)

  # normally you would do some training here
  # but fornow we will just assign something to W
  sess.run(tf.assign(W, [[1, 2],[4,5],[7,8]]))
  sess.run(tf.assign(b, [1,1]))

  builder.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.TRAINING],
                                       signature_def_map=None,
                                       assets_collection=None)
builder.save()
