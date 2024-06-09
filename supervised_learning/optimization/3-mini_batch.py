#!/usr/bin/env python3

"""This module contains the function that
that trains a loaded neural network model using
mini-batch gradient descent:
"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model using mini-batch gradient descent
    X_train - is a numpy.ndarray of shape (m, 784) containing the training data
        m - is the number of data points
        784 - is the number of input features
    Y_train - is a one-hot of shape (m, 10) containing the training labels
        10 - is the number of classes the model should classify
    X_valid - is a of shape (m, 784) with validation data
    Y_valid - is a one-hot of shape (m, 10) containing the validation labels
    batch_size - is the number of data points in a batch
    epochs - is the number of times the training should pass
    through the whole dataset
    load_path - is the path from which to load the model
    save_path - is the path to where the model should be saved after training
    Returns: the path where the model was saved"""

    # Shuffle the training data
    X_train, Y_train = shuffle_data(X_train, Y_train)

    # Load the model
    session = tf.Session()
    saver = tf.train.import_meta_graph(load_path + '.meta')
    saver.restore(session, load_path)

    # Access the graph
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    accuracy = graph.get_tensor_by_name("Mean_1:0")
    loss = graph.get_tensor_by_name("Mean:0")
    train_op = graph.get_operation_by_name("train_op")

    # Calculate the number of batches for training and validation
    m = X_train.shape[0]
    num_batches = m // batch_size if m % batch_size == 0 else (
        m // batch_size) + 1

    # Training loop
    for epoch in range(epochs):
        # Shuffle data at the start of each epoch
        X_train, Y_train = shuffle_data(X_train, Y_train)
        for i in range(num_batches):
            start_i = i * batch_size
            end_i = start_i + batch_size
            X_mini = X_train[start_i:end_i]
            Y_mini = Y_train[start_i:end_i]
            session.run(train_op, feed_dict={x: X_mini, y: Y_mini})

        # Calculate loss and accuracy for training and validation sets
        cost_train, acc_train = session.run(
            [loss, accuracy], feed_dict={x: X_train, y: Y_train})
        cost_valid, acc_valid = session.run(
            [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

    print("After epoch {}/{}:".format(epoch + 1, epochs))
    print("\tTraining Cost: {}".format(cost_train))
    print("\tTraining Accuracy: {}".format(acc_train))
    print("\tValidation Cost: {}".format(cost_valid))
    print("\tValidation Accuracy: {}".format(acc_valid))

    # Save the model
    saved_path = saver.save(session, save_path)
    session.close()

    return saved_path
