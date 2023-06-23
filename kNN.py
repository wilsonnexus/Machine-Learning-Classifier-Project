# Author: Wilson Neira

class c_k_NN_algorithm:
    """
    A class for implementing the k-Nearest Neighbor algorithm.

    Args:
    x_train (list): List of training data features.
    y_train (list): List of training data labels.
    k (int): Number of neighbors to consider for k-NN.

    Attributes:
    x_train (list): List of training data features.
    y_train (list): List of training data labels.
    k (int): Number of neighbors to consider for k-NN.
    """

    def __init__(self, x_train, y_train, k):
        """
        Constructor to initialize training data features, labels, and number of neighbors.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    def euclidean_distance(self, x_test):
        """
        Calculate Euclidean distance between training and testing data.

        Args:
        x_test (list): List of test data features.

        Returns:
        list: List of Euclidean distances between each test sample and all training samples.
        """
        test_dists = []
        for i in range(len(x_test)):
            dists = [
                (sum([(float(x_test[i][j]) - float(self.x_train[m][j])) ** 2 for j in range(len(x_test[i]))])) ** 0.5
                for m in range(len(self.x_train))]
            test_dists.append(dists)
        return test_dists

    def test(self, x_test):
        """
        Test the model on the test data and make predictions.

        Args:
        x_test (list): List of test data features.

        Returns:
        list: List of predicted labels for the test data.
        """
        y_test_pred = []
        test_dists = self.euclidean_distance(x_test)  # Find Euclidean Distance between training and test data
        for dists in test_dists:
            k_smallest = sorted(range(len(dists)), key=lambda i: dists[i])[:self.k]
            labels = [self.y_train[i] for i in k_smallest]
            label_counts = {}
            for label in labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
            y_test_pred.append(max(label_counts, key=label_counts.get))

        return y_test_pred
