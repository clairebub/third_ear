#!/usr/bin/env python

import argparse
import tensorflow as tf

import urban_sound_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int,
                    help='batch size')
parser.add_argument('--train_steps', default=20, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = urban_sound_data.load_data(
            train_set_percent=0.8)
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
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=len(urban_sound_data.SOUND_CLASSES),
        model_dir='models/urban_sound',
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

    # Generate predictions from the model
    (x, y), (_, _) = urban_sound_data.load_data(
            train_set_percent=1)
    #expected = ['1', '2', '1']
    #predict_x = {
    #    'SepalLength': [5.1, 5.9, 6.9],
    #    'SepalWidth': [3.3, 3.0, 3.1],
    #    'PetalLength': [1.7, 4.2, 5.4],
    #    'PetalWidth': [0.5, 1.5, 2.1],
    #}
    #print("x\n", x)
    #print("y\n", y)
    y_preds = classifier.predict(
        input_fn=lambda:urban_sound_data.eval_input_fn(
                x,
                labels=None,
                batch_size=args.batch_size))
    for y_pred, y in zip(y_preds, y['y']):
        class_id = y_pred['class_ids'][0]
        probability = y_pred['probabilities'][class_id]
        print("Expected %s, prediction is %s with softmax at %.3f."%(
                    urban_sound_data.SOUND_CLASSES[y],
                    urban_sound_data.SOUND_CLASSES[class_id],
                    100 * probability,))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
