import numpy as np
import math
from sklearn.datasets import make_circles


def two_moons_dataset(size, training_split):
    """
    Creates a 2 moons dataset (corresponds to 'dataset 5' from lecture)
    :param size: number of total samples within the dataset
    :param training_split: ratio of samples used for training
    :return: dataset split into a training and a test set
    """
    # split size in half
    size1 = math.floor(size / 2)
    size2 = math.ceil(size / 2)

    # create first banana
    x1_banana1 = np.random.uniform(low=-2, high=6, size=size1)
    x2_banana1 = -0.3 * (x1_banana1-2)**2 + 4 + np.random.randn(size1) * 0.7

    # create second banana
    x1_banana2 = np.random.uniform(low=-6,  high=2, size=size2)
    x2_banana2 = 0.3 * (x1_banana2+2)**2 - 4 + np.random.randn(size2) * 0.7

    # combine x1 and x2
    data1 = np.stack((x1_banana1, x2_banana1), axis=1)
    data2 = np.stack((x1_banana2, x2_banana2), axis=1)

    # create labels
    labels1 = np.zeros_like(x1_banana1)
    labels2 = np.ones_like(x1_banana2)

    # split dataset in train and test data
    n_train1 = round(training_split * size1)
    n_train2 = round(training_split * size2)
    x_train = np.concatenate((data1[:n_train1, :], data2[:n_train2, :]))
    y_train = np.concatenate((labels1[:n_train1], labels2[:n_train2]))
    x_test = np.concatenate((data1[n_train1:, :], data2[n_train2:, :]))
    y_test = np.concatenate((labels1[n_train1:], labels2[n_train2:]))

    return x_train, y_train, x_test, y_test


def four_parallel_dataset(size, training_split):
    """
    Creates a dataset consisting of four parallel distributions (corresponds to 'dataset 4' from lecture)
    :param size: number of total samples within the dataset
    :param training_split: ratio of samples used for training
    :return: dataset split into a training and a test set
    """
    # divide size by four
    size1 = math.floor(size / 4)
    size2 = math.ceil(size / 4)
    size3 = math.floor(size / 4)
    size4 = math.ceil(size / 4)

    # create distributions
    x1_dist1 = np.random.randn(size1)
    x2_dist1 = np.random.uniform(low=-1, high=1, size=size1)
    x1_dist2 = np.random.randn(size2) + 4
    x2_dist2 = np.random.uniform(low=-1, high=1, size=size2)
    x1_dist3 = np.random.randn(size3) + 8
    x2_dist3 = np.random.uniform(low=-1, high=1, size=size3)
    x1_dist4 = np.random.randn(size4) + 12
    x2_dist4 = np.random.uniform(low=-1, high=1, size=size4)

    # combine x1 and x2
    data1 = np.stack((x1_dist1, x2_dist1), axis=1)
    data2 = np.stack((x1_dist2, x2_dist2), axis=1)
    data3 = np.stack((x1_dist3, x2_dist3), axis=1)
    data4 = np.stack((x1_dist4, x2_dist4), axis=1)

    # create labels
    labels1 = np.zeros_like(x1_dist1)
    labels2 = np.ones_like(x1_dist2)
    labels3 = np.zeros_like(x1_dist3)
    labels4 = np.ones_like(x1_dist4)

    # split dataset in train and test data
    n_train1 = round(training_split * size1)
    n_train2 = round(training_split * size2)
    n_train3 = round(training_split * size3)
    n_train4 = round(training_split * size4)
    x_train = np.concatenate((data1[:n_train1, :], data2[:n_train2, :], data3[:n_train3, :], data4[:n_train4, :]))
    y_train = np.concatenate((labels1[:n_train1], labels2[:n_train2], labels3[:n_train3], labels4[:n_train4]))
    x_test = np.concatenate((data1[n_train1:, :], data2[n_train2:, :], data3[n_train3:, :], data4[n_train4:, :]))
    y_test = np.concatenate((labels1[n_train1:], labels2[n_train2:], labels3[n_train3:], labels4[n_train4:]))

    return x_train, y_train, x_test, y_test


def four_gaussian_dataset(size, training_split):
    """
    Creates a dataset consisting of four Gaussian distributions
    :param size: number of total samples within the dataset
    :param training_split: ratio of samples used for training
    :return: dataset split into a training and a test set
    """
    # divide size by four
    size1 = math.floor(size / 4)
    size2 = math.ceil(size / 4)
    size3 = math.floor(size / 4)
    size4 = math.ceil(size / 4)

    # create Gaussian distributions
    data1 = np.random.randn(size1, 2) + np.array([2, 2])
    data2 = np.random.randn(size2, 2) + np.array([2, -2])
    data3 = np.random.randn(size3, 2) - np.array([2, 2])
    data4 = np.random.randn(size4, 2) - np.array([2, -2])

    # create labels
    labels1 = np.zeros_like(data1[:, 0])
    labels2 = np.ones_like(data2[:, 0])
    labels3 = np.zeros_like(data3[:, 0])
    labels4 = np.ones_like(data4[:, 0])

    # split dataset in train and test data
    n_train1 = round(training_split * size1)
    n_train2 = round(training_split * size2)
    n_train3 = round(training_split * size3)
    n_train4 = round(training_split * size4)
    x_train = np.concatenate((data1[:n_train1, :], data2[:n_train2, :], data3[:n_train3, :], data4[:n_train4, :]))
    y_train = np.concatenate((labels1[:n_train1], labels2[:n_train2], labels3[:n_train3], labels4[:n_train4]))
    x_test = np.concatenate((data1[n_train1:, :], data2[n_train2:, :], data3[n_train3:, :], data4[n_train4:, :]))
    y_test = np.concatenate((labels1[n_train1:], labels2[n_train2:], labels3[n_train3:], labels4[n_train4:]))

    return x_train, y_train, x_test, y_test


def circular_dataset(size, training_split):
    """
    Creates a circular dataset (corresponds to 'dataset 2' from lecture)
    :param size: number of total samples within the dataset
    :param training_split: ratio of samples used for training
    :return: dataset split into a training and a test set
    """
    # split size in half
    size1 = math.floor(size / 2)
    size2 = math.ceil(size / 2)

    # create circular dataset
    data, labels = make_circles(size, factor=0, shuffle=False)
    data = data * 4 + np.random.randn(size, 2)

    # split dataset in train and test data
    data1 = data[:size1, :]
    data2 = data[size1:, :]
    labels1 = labels[:size1]
    labels2 = labels[size1:]
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    n_train1 = round(training_split * size1)
    n_train2 = round(training_split * size2)
    x_train = np.concatenate((data1[:n_train1, :], data2[:n_train2, :]))
    y_train = np.concatenate((labels1[:n_train1], labels2[:n_train2]))
    x_test = np.concatenate((data1[n_train1:, :], data2[n_train2:, :]))
    y_test = np.concatenate((labels1[n_train1:], labels2[n_train2:]))

    return x_train, y_train, x_test, y_test
