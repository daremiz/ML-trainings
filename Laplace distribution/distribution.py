import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        # Calculate the median for each feature
        medians = np.median(x, axis=0)

        # Calculate the mean absolute deviation from the median for each feature
        mad = np.mean(np.abs(x - medians), axis=0)

        return mad

    def __init__(self, features):
        '''
        Args:
        feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        # Calculate the location parameter (median) and scale parameter based on the input features
        self.loc = np.median(features, axis=0)
        self.scale = np.mean(np.abs(features - self.loc), axis=0)

    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        # Calculate the log probability density function for Laplace distribution
        log_prob = -np.log(2 * self.scale) - np.abs(values - self.loc) / self.scale
        return log_prob

    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        # Calculate the probability density function for Laplace distribution
        prob = np.exp(self.logpdf(values))
        return prob

