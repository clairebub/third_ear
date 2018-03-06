#!/usr/bin/env python

import argparse
import tensorflow as tf

import urban_sound_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int,
                    help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

#feature_spec = {'foo': tf.FixedLenFeature(...),
#                'bar': tf.VarLenFeature(...)}
#
#def serving_input_receiver_fn():
#  """An input receiver that expects a serialized tf.Example."""
#  serialized_tf_example = tf.placeholder(dtype=tf.string,
#                                         shape=[default_batch_size],
#                                         name='input_example_tensor')
#  receiver_tensors = {'examples': serialized_tf_example}
#  features = tf.parse_example(serialized_tf_example, feature_spec)
#  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def main(argv):
    args = parser.parse_args(argv[1:])

    train_x, train_y = urban_sound_data.load_data([1, 2, 3, 4, 5, 6, 7, 8])
    print("train shapes", train_x.shape, train_y.shape)
    test_x, test_y = urban_sound_data.load_data([9])
    print("test shapes", test_x.shape, test_y.shape)
    #print("train_x\n", train_x)
    #print("train_y\n", train_y)
    #print("test_x\n", test_x)
    #print("test_y\n", test_y)
    # Feature columns describe how to use the input.
    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs = 20*60,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max = 3,       # Retain the 10 most recent checkpoints.
    )
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[320, 320],
        # The model must choose between 3 classes.
        n_classes=len(urban_sound_data.SOUND_CLASSES),
        model_dir='./model/urban_sound/',
        config=checkpointing_config)

    # Train the Model.
    classifier.train(
        input_fn=lambda:urban_sound_data.train_input_fn(
                train_x,
                train_y,
                args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:urban_sound_data.eval_input_fn(
                test_x,
                test_y,
                args.batch_size))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # export the trained model
    export_dir_base = './model_export'
#    serving_input_receiver_fn
#    classifier.export_savedmodel()

    # Generate predictions from the model
    x, y = urban_sound_data.load_data([9])
    y_preds = classifier.predict(
        input_fn=lambda:urban_sound_data.eval_input_fn(
                x,
                labels=None,
                batch_size=args.batch_size))
    infer_count_accurate = {}
    infer_count_total = {}
    for i in range(len(urban_sound_data.SOUND_CLASSES)):
        infer_count_total[i] = 0
        infer_count_accurate[i] = 0

    for y_pred, y in zip(y_preds, y['y']):
        class_id = y_pred['class_ids'][0]
        probability = y_pred['probabilities'][class_id]
        infer_count_total[y] += 1
        if y == class_id:
            infer_count_accurate[y] += 1
#        print("Expected %s, prediction is %s with softmax at %.3f."%(
#                    urban_sound_data.SOUND_CLASSES[y],
#                    urban_sound_data.SOUND_CLASSES[class_id],
#                    100 * probability,))
    for i in range(len(urban_sound_data.SOUND_CLASSES)):
        print("urban sound %s: accurate=%d, total=%d, accuracy rate=%.3f" %
                (urban_sound_data.SOUND_CLASSES[i],
                infer_count_accurate[i],
                infer_count_total[i],
                infer_count_accurate[i]/infer_count_total[i]))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
