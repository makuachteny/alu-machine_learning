#!/usr/bin/env python3

"""This module contains the function that
builds, trains, and saves a neural network model in tensorflow
using Adam optimization, mini-batch gradient descent,
learning rate decay, and batch normalization:
"""

import tensorflow as tf
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    builds, trains, and saves a neural network model in tensorflow
    using Adam optimization, mini-batch gradient descent,
    learning rate decay, and batch normalization
    Data_train - is a tuple containing the training inputs and training labels
    Data_valid - is a tuple containing the validation inputs and validation labels
    layers - is a list containing the number of nodes in each layer of the network
    activations - is a list containing the activation functions for each layer of the network
    alpha - is the learning rate
    beta1 - is the weight used for the first moment
    beta2 - is the weight used for the second moment
    epsilon - is a small number to avoid division by zero
    decay_rate - is the decay rate for inverse time decay of the learning rate
    batch_size - is the number of data points in a batch
    epochs - is the number of times the training should pass through the whole dataset
    save_path - is the path where the model should be saved
    Returns: the path where the model was saved
    """

    # Unpack the data
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # Create placeholders for the input data
    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='x')
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]), name='y')

    # Create the neural network
    A = create_layer(x, layers[0], activations[0])
    for i in range(1, len(layers)):
        A = create_batch_norm_layer(A, layers[i], activations[i])
    y_pred = A

    # Create the loss function
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # Create the learning rate decay operation
    global_step = tf.Variable(0, trainable=False)
    alpha = tf.train.inverse_time_decay(
        alpha, global_step, decay_rate, 1, staircase=True)

    # Create the Adam optimization operation
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Create the accuracy operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create the saver
    saver = tf.train.Saver()

    # Create the session
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs + 1):
            # Shuffle the training data
            X_train, Y_train = shuffle_data(X_train, Y_train)
            # Train the model
            session.run(train_op, feed_dict={x: X_train, y: Y_train})
            # Validate the model
            if epoch % 10 == 0:
                train_cost = session.run(
                    loss, feed_dict={x: X_train, y: Y_train})
                valid_cost = session.run(
                    loss, feed_dict={x: X_valid, y: Y_valid})
                train_accuracy = session.run(
                    accuracy, feed_dict={x: X_train, y: Y_train})
                valid_accuracy = session.run(
                    accuracy, feed_dict={x: X_valid, y: Y_valid})
                print("After {} epochs:".format(epoch))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))
        # Save the model
        save_path = saver.save(session, save_path)
    return save_path
