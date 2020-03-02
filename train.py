"""
    Deep Neural Network for Digit Recognition Dataset from Kaggle
    Accuracy in test dataset: 91.4%

    Author: Rennan Rocha
"""

import numpy as np
import matplotlib.pyplot as plt
import json

from read_dataset import read_train, read_test
from sklearn.metrics import precision_recall_fscore_support

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 32
    n_y = Y.shape[0]
    return n_x, n_h, n_y

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = Z1 * (Z1 > 0)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    assert(A2.shape == (W2.shape[0], X.shape[1]))
    return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[1]

    logprobs = ((Y * np.log(A2)) + ((1 - Y) * np.log(1 - A2))).sum(axis = 1, keepdims = True) / m
    cost = -1/logprobs.shape[0] * logprobs.sum()

    cost = float(np.squeeze(cost))

    assert(isinstance(cost, float))

    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * dZ2.sum(axis = 1, keepdims = True)

    dA1 = A1
    dA1[dA1 < 0] = 0
    dA1[dA1 >= 0] = 1
    dZ1 = np.dot(W2.T, dZ2) * dA1
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims=True)


    assert(dW1.shape == W1.shape)
    assert(dW2.shape == W2.shape)
    assert(db1.shape == b1.shape)
    assert(db2.shape == b2.shape)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }
    return grads

def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def predict(X, parameters):
    m = X.shape[1]
    A2, _ = forward_propagation(X, parameters)
    return A2.argmax(axis = 0).reshape((m, 1))

def nn_model(X, Y, num_iterations = 2000, learning_rate = 0.005, print_cost = False):
    n_x, n_h, n_y = layer_sizes(X, Y)

    parameters = initialize_parameters(n_x, n_h, n_y)

    costs = np.zeros((1, num_iterations))

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        costs[0, i] = cost

    return parameters, costs

def main():
    print("Reading dataset...")
    X_train, Y_train, Y_train_raw = read_train()
    print("Dataset ready.")

    print("Start training...")
    learning_rate = 0.045
    num_iterations = 2500
    parameters, costs = nn_model(X_train, Y_train, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
    print("Traning finished.")

    print("Predicting train set...")
    Y_prediction_train = predict(X_train, parameters)
    precision, recall, fscore, _ = precision_recall_fscore_support(Y_train_raw.reshape((X_train.shape[1], 1)), Y_prediction_train, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print("Precistion:", precision)
    print("Recall:", recall)
    print("Fscore:", fscore)

    print("Reading test set...")
    X_test = read_test()
    print("Predicting test set...")
    Y_prediction_test = predict(X_test, parameters)
    print("Predicted.")

    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iterations")
    plt.title("Learning rate = " +  str(learning_rate))
    plt.show()

    d = {
        "costs": costs.tolist(),
        "Y_prediction_test": Y_prediction_test.tolist(),
        "Y_prediction_train": Y_prediction_train.tolist(),
        "W1": parameters["W1"].tolist(),
        "W2": parameters["W2"].tolist(),
        "b1": parameters["b1"].tolist(),
        "b2": parameters["b2"].tolist(),
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    print("Saving model...")
    jd = json.dumps(d)
    with open("model.json", "w") as f:
        f.write(jd)
    print("Model saved.")

if __name__ == '__main__':
    main()