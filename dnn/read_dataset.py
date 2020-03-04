import csv
import numpy as np

def read_train():
    with open('../datasets/train.csv', 'r') as f:
        data_iter = csv.reader(f, delimiter=',')
        data = [data for data in data_iter]

    train_arr = np.asarray(data, dtype=None)

    m_train = train_arr.shape[0] - 1
    n_train = train_arr.shape[1] - 1

    x_train_set = np.asarray(train_arr[1:, 1:], dtype=np.float).T.reshape((n_train, m_train)) / 255.

    y_train_raw = np.asarray(train_arr[1:, 0], dtype=np.uint).reshape((1, m_train))
    y_train_set = np.zeros((10, m_train))
    y_train_set[y_train_raw[0, :].reshape(1, m_train), np.arange(m_train)] = 1
    print(y_train_set.sum(axis = 1))

    assert(x_train_set.shape[0] == n_train)
    assert(x_train_set.shape[1] == m_train)
    assert(y_train_set.shape[1] == m_train)

    return x_train_set, y_train_set, y_train_raw

def read_test():
    with open('../datasets/test.csv', 'r') as f:
        data_iter = csv.reader(f, delimiter=',')
        data = [data for data in data_iter]

    test_arr = np.asarray(data, dtype=None)
    n_test = test_arr.shape[1]
    m_test = test_arr.shape[0] - 1
    x_test_set = np.asarray(test_arr[1:, :], dtype=np.float).T.reshape((n_test, m_test)) / 255.
    return x_test_set
