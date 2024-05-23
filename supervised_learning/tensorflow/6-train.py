#!/usr/bin/env python3
""" Module builds, trains, and saves a neural network """
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """ Builds, trains, and saves a neural network """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    
    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)
    
    # Loss
    loss = calculate_loss(y_pred, y)
    
    # Training
    train_op = create_train_op(loss, alpha)
    
    # Accuracy
    accuracy = calculate_accuracy(y_pred, y)
    
    # Initialize all variables
    init = tf.global_variables_initializer()
    
    # Add to the graph
    tf.compat.v1.add_to_collection('x', x)
    tf.compat.v1.add_to_collection('y', y)
    tf.compat.v1.add_to_collection('y_pred', y_pred)
    tf.compat.v1.add_to_collection('loss', loss)
    tf.compat.v1.add_to_collection('train_op', train_op)
    tf.compat.v1.add_to_collection('accuracy', accuracy)
    
    # Save model
    saver = tf.train.Saver()
    
    # Start the session
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(iterations + 1):
            feed_dict_train = {x: X_train, y: Y_train}
            feed_dict_valid = {x: X_valid, y: Y_valid}
            
            if i % 100 == 0 or i == iterations:
                t_cost, t_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict_train)
                v_cost, v_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(t_cost))
                print("\tTraining Accuracy: {}".format(t_accuracy))
                print("\tValidation Cost: {}".format(v_cost))
                print("\tValidation Accuracy: {}".format(v_accuracy))
            
            if i < iterations:
                sess.run(train_op, feed_dict=feed_dict_train)

        saver.save(sess, save_path)
        
    return save_path
 