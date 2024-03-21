import numpy as np
from sklearn.mixture import GaussianMixture


class NearestMeanClassifier:
    def __init__(self):
        self.means = np.array([[0, 0], [0, 0]])

    def fit(self, x_train, y_train):
        # calculate means
        x_train0 = x_train[y_train == 0]
        mean0 = np.mean(x_train0, axis=0)
        x_train1 = x_train[y_train == 1]
        mean1 = np.mean(x_train1, axis=0)
        self.means = np.stack((mean0, mean1))

    def predict(self, x_test):
        labels = np.zeros(len(x_test), dtype=np.int32)

        for i, x in enumerate(x_test):
            # calculate distances of test instance x to both means
            distances = np.sum((self.means - x) ** 2, axis=1)
            # identify nearest mean
            label = np.argmin(distances)
            # save the label in array which should finally contain the labels for all test instances
            labels[i] = label

        return labels


class KNearestNeighborClassifier:
    def __init__(self, k):
        self.k = k
        self.x_train = np.array([])
        self.y_train = np.array([])

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        labels = np.zeros(len(x_test), dtype=np.int32)

        for i, x in enumerate(x_test):
            # calculate distances of test instance x to all training instances
            distances = np.sum((self.x_train - x) ** 2, axis=1)

            # sort the labels of the training instances according to their distance to x
            sorted_labels = self.y_train[np.argsort(distances)]

            # take first k labels
            knn_labels = sorted_labels[:self.k].astype(np.int32)

            # identify the most frequently occurring label
            label = np.argmax(np.bincount(knn_labels))

            # save the label in array which should finally contain the labels for all test instances
            labels[i] = label

        return labels


class GaussianMixtureModelClassifier:
    def __init__(self, m):
        self.gmm0 = GaussianMixture(m[0])
        self.gmm1 = GaussianMixture(m[1])

    def fit(self, x_train, y_train):
        # fit one Gaussian mixture model to each class
        x_train0 = x_train[y_train == 0]
        self.gmm0.fit(x_train0)
        x_train1 = x_train[y_train == 1]
        self.gmm1.fit(x_train1)

    def predict(self, x_test):
        # maximum likelihood classification
        log_likelihood0 = self.gmm0.score_samples(x_test)
        log_likelihood1 = self.gmm1.score_samples(x_test)
        log_likelihood_ratio = log_likelihood1 - log_likelihood0
        labels = np.zeros_like(x_test[:, 0])
        labels[log_likelihood_ratio > 0] = 1

        return labels
