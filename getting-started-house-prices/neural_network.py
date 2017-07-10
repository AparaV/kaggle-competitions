import random
import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf

from cleaning import cleanup, encode_features, split

# Change these variables accordingly
data_dir = 'data'
num_steps = 1000001

def clean(train_dataset, test_dataset):
    train_dataset = cleanup(train_dataset)
    test_dataset = cleanup(test_dataset)
    train_dataset, test_dataset = encode_features(train_dataset, test_dataset)

    x_train, y_train, x_test, y_test, x_valid, y_valid = split(train_dataset)

    return x_train, y_train, x_test, y_test, x_valid, y_valid, test_dataset

def accuracy(prediction, labels):
    return 0.5 * np.sqrt(((prediction - labels) ** 2).mean(axis=None))

def main():
    train_dataset = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_dataset = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    x_train, y_train, x_test, y_test, x_valid, y_valid, test_dataset = clean(train_dataset, test_dataset)

    train_size = np.shape(x_train)[0]
    valid_size = np.shape(x_valid)[0]
    test_size = np.shape(x_test)[0]
    num_features = np.shape(x_train)[1]
    num_hidden = 16 # Number of activation units in the hidden layer

    # Neural network with one hidden layer
    graph = tf.Graph()
    with graph.as_default():

        # Input
        tf_train_dataset = tf.constant(x_train, dtype=tf.float32)
        tf_train_labels = tf.constant(y_train, dtype=tf.float32)
        tf_valid_dataset = tf.constant(x_valid)
        tf_test_dataset = tf.constant(x_test)

        # Variables
        weights_1 = tf.Variable(tf.truncated_normal(
            [num_features, num_hidden]), dtype=tf.float32, name="layer1_weights")
        biases_1 = tf.Variable(tf.zeros([num_hidden]), dtype=tf.float32, name="layer1_biases")
        weights_2 = tf.Variable(tf.truncated_normal(
            [num_hidden, 1]), dtype = tf.float32, name="layer2_weights")
        biases_2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="layer2_biases")
        steps = tf.Variable(0)

        # Model
        def model(x, train=False):
            hidden = tf.nn.relu(tf.matmul(x, weights_1) + biases_1)
            return tf.matmul(hidden, weights_2) + biases_2

        # Loss Computation
        train_prediction = model(tf_train_dataset)
        loss = 0.5 * tf.reduce_mean(tf.squared_difference(tf_train_labels, train_prediction))
        cost = tf.sqrt(loss)

        # Optimizer
        # Exponential decay of learning rate
        learning_rate = tf.train.exponential_decay(0.06, steps, 5000, 0.70, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=steps)

        # Predictions
        valid_prediction = model(tf_valid_dataset)
        test_prediction = model(tf_test_dataset)

        saver = tf.train.Saver()

    # Running graph
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, c, predictions = sess.run([optimizer, cost, train_prediction])
            if (step % 5000 == 0):
                print('Cost at step %d: %.2f' % (step, c))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                print('Validation loss: %.2f' % accuracy(valid_prediction.eval(), y_valid))
        t_pred = test_prediction.eval()
        print('Test loss: %.2f' % accuracy(t_pred, y_test))
        save_path = saver.save(sess, "./model/nn-model.ckpt")
        print('Model saved in %s' % (save_path))

    # Reconstructing graph and predicting outputs
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, "./model/nn-model.ckpt")
        print("Model restored.\nMaking predictions...")
        x = test_dataset.drop('Id', axis=1).as_matrix().astype(dtype=np.float32)
        hidden = tf.nn.relu(tf.matmul(x, weights_1) + biases_1)
        y = tf.cast((tf.matmul(hidden, weights_2) + biases_2), dtype=tf.uint16).eval()
        test_dataset['SalePrice'] = y
        output = test_dataset[['Id', 'SalePrice']]

    output.to_csv('./submissions/nn-submission.csv', index=False)
    print("Output saved to ./submissions/nn-submission.csv")

if __name__ == "__main__":
    main():
