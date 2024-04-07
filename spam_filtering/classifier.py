from absl import logging
import time

import numpy as np

class SpamClassifier(object):
    """
    Spam classifier using (multinomial) Naive Bayes

    Parameters:
        alpha (float): Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    """
    def __init__(self, alpha=1.0):
        super(SpamClassifier, self).__init__()
        self.alpha = alpha

    def train(self, X, y):
        """
        Training method

        Estimates the log-likelihoods and priors for both classes ham and spam.

        Parameters:
            X (ndarray): Feature matrix with shape (num_samples, num_features)
            y (ndarray): Label vector with shape (num_samples,)
        """
        logging.info(f"Starting training...")
        start_time = time.time()

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        self.priors = np.zeros((self.n_classes,)) # This is just a placeholder
        for i in self.classes:
            self.priors[i] = (y == i).sum() / n_samples
        # TODO: Estimate priors
        self.log_priors = np.log(self.priors)

        self.log_probs = np.zeros((self.n_classes, n_features))
        # TODO: Estimate log-likelihoods
        for i in self.classes:
            self.log_probs = np.log((np.sum(X[y == i], axis=0) + self.alpha) / np.sum(X[y == i]) + self.alpha * n_features)

        logging.debug(f"Training took {int(time.time() - start_time)} seconds.")


    def predict(self, X):
        """
        Prediction method

        Uses Bayes rule to compute un-normalized posteriors

        Parameters:
            X (ndarray): Feature matrix with shape (num_samples, num_features)

        Returns:
            (ndarray): Prediction vector with shape (num_samples,)
        """
        # TODO: Implement MAP decision for multinomial Naive Bayes
        posteriors =
        return np.zeros(X.shape[0]) # This is just a placeholder
        






