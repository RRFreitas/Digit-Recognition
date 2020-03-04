"""
    Convolutional Neural Network for digite recognition dataset from Kaggle
    Accuracy in test dataset: 98.04%

    Author: Rennan Rocha
"""

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import json
from cnn_utils import *
from tensorflow.python.framework import ops
tf.disable_v2_behavior()

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, (None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, (None, n_y))
    return X, Y

def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [4, 4, 1, 8])
    W2 = tf.get_variable("W2", [2, 2, 8, 16])

    parameters = {"W1": W1,
                  "W2": W2}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    F = tf.layers.flatten(P2)
    Z3 = tf.layers.dense(F, units=10, activation=None)

    return Z3

def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost

def model(X_train, Y_train, X_test, learning_rate=0.009,
          num_epochs = 100, mini_batch_size = 64, print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / mini_batch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, mini_batch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _, temp_cost = sess.run(
                        fetches=[optimizer, cost],
                        feed_dict={
                            X: minibatch_X,
                            Y: minibatch_Y
                        }
                )

                minibatch_cost += temp_cost / num_minibatches

            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost:
                costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title('Learning rate =' + str(learning_rate))
        plt.show()

        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        print("Train Accuracy:", train_accuracy)

        test_predict = predict_op.eval({X: X_test})

    return train_accuracy, test_predict, parameters

def main():
    x_train_set, y_train_set, y_train_raw = read_train()
    x_test_set = read_test()

    print("number of training examples =", x_train_set.shape[0])
    print("number of test examples =", x_test_set.shape[0])
    print("X_train shape:", x_train_set.shape)
    print("Y_train shape:", y_train_set.shape)
    print("X_train shape:", x_test_set.shape)

    train_accuracy, test_predict, parameters = model(x_train_set, y_train_set, x_test_set, num_epochs=25)
    p = {
        "predict": test_predict.tolist(),
    }
    print("Saving model...")
    jd = json.dumps(p)
    with open("model.json", "w") as f:
        f.write(jd)
    print("Model saved.")


if __name__ == '__main__':
    main()