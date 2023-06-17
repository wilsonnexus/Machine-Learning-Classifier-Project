# Author: Wilson Neira
# Evaluating the k-NN Algorithm
# Evaluating the Decision Tree Algorithm
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" k-Nearest Neighbors (k-NN) Algorithm Begins Here"""
class c_k_NN_algorithm:

    """
    A class for implementing the k-Nearest Neighbor algorithm.
    """

    def __init__(self, x_train, y_train, k):
        """
        Train the data.

        :param x_train: Training data features.
        :param y_train: Training data labels.
        :param k (int): Number of considered neighbors

        :return: None
        """
        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    def euclidean_distance(self, x_test):
        """
        Caclulate euclidean distance between training and testing data.
        :param x_test (list): Test data features.
        :return test_dists (list): List of euclidean distances between training and testing data.
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
        Test the data.
        :param x_test (list): Test data features.
        :return y_test_pred (list): List od predicted labels for the test data.
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
