import numpy as np
import math
import csv

def read_train():
    with open('../datasets/train.csv', 'r') as f:
        data_iter = csv.reader(f, delimiter=',')
        data = [data for data in data_iter]

    train_arr = np.asarray(data, dtype=None)

    m_train = train_arr.shape[0] - 1
    n_pixels = 28

    x_train_set = np.asarray(train_arr[1:, 1:], dtype=np.float).reshape((m_train, n_pixels, n_pixels, 1)) / 255.

    y_train_raw = np.asarray(train_arr[1:, 0], dtype=np.uint).reshape((m_train, 1))
    y_train_set = np.zeros((m_train, 10))
    y_train_set[np.arange(m_train), y_train_raw[:, 0].reshape(1, m_train)] = 1

    print(y_train_set.sum(axis = 0))

    assert(x_train_set.shape[0] == m_train)
    assert(x_train_set.shape[1] == n_pixels)
    assert(x_train_set.shape[2] == n_pixels)
    assert(x_train_set.shape[3] == 1)
    assert(y_train_set.shape[0] == m_train)

    return x_train_set, y_train_set, y_train_raw

def read_test():
    with open('../datasets/test.csv', 'r') as f:
        data_iter = csv.reader(f, delimiter=',')
        data = [data for data in data_iter]

    test_arr = np.asarray(data, dtype=None)
    n_pixels = 28
    m_test = test_arr.shape[0] - 1
    x_test_set = np.asarray(test_arr[1:, :], dtype=np.float).reshape((m_test, n_pixels, n_pixels, 1)) / 255.
    return x_test_set

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
