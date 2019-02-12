import numpy as np


def sample_decision_tree_data():
    features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['c', 'b']]
    labels = [0, 0, 1, 1]
    return features, labels


def sample_decision_tree_test():
    features = [['a', 'b'], ['b', 'a'], ['b', 'b']]
    labels = [0, 0, 0]
    return features, labels


def load_decision_tree_data():
    f = open('car.data', 'r')
    white = [[int(num) for num in line.split(',')] for line in f]
    white = np.asarray(white)

    [N, d] = white.shape

    ntr = int(np.round(N * 0.66))
    ntest = N - ntr

    Xtrain = white[:ntr].T[:-1].T
    ytrain = white[:ntr].T[-1].T
    Xtest = white[-ntest:].T[:-1].T
    ytest = white[-ntest:].T[-1].T

    return Xtrain, ytrain, Xtest, ytest




