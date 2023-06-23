# Author: Wilson Neira
# Evaluating the Different Classifier Ensemble
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import random_forest
import kNN
import RadiusNeighborsClassifier


class Ensemble:
    """
    A class for implementing an Ensemble Classifier, which aggregates the outputs of multiple classifiers.

    Args:
    S_data_train (pandas DataFrame): Training data.
    C_labels (list): List of labels for the features.
    criteria (str): The type of algorithm to use for splitting (either "InfoGain" or "Gini").
    simple (bool): Indicator to decide if it's a simple decision tree.
    WeakLearn (bool): Weak learning parameter for the Random Forest.
    ntree (int): Number of trees in the Random Forest.
    F (float): The fraction of the training data to be used for creating each bootstrap sample.
    kNN (bool): Indicator to decide if k-Nearest Neighbors is used.
    kNNk (int): Number of neighbors to consider in k-Nearest Neighbors.
    radius (float): The radius within which to look for neighbors in Radius Neighbors Classifier.

    Attributes:
    C_labels (list): List of labels for the features.
    criteria (str): The type of algorithm to use for splitting.
    simple (bool): Indicator to decide if it's a simple decision tree.
    WeakLearn (bool): Weak learning parameter for the Random Forest.
    ntree (int): Number of trees in the Random Forest.
    F (float): The fraction of the training data to be used for creating each bootstrap sample.
    kNN (bool): Indicator to decide if k-Nearest Neighbors is used.
    kNNk (int): Number of neighbors to consider in k-Nearest Neighbors.
    radius (float): The radius within which to look for neighbors in Radius Neighbors Classifier.
    out_of_bag (list): List of out-of-bag samples.
    ensemble (list): List of classifiers in the ensemble.
    """
    def __init__(self, S_data_train, C_labels, criteria, simple, WeakLearn, ntree, F, kNN, kNNk, radius):
        """
        Initialize a Ensemble.
        """
        self.C_labels = C_labels
        self.criteria = criteria
        self.simple = simple
        self.WeakLearn = WeakLearn
        self.ntree = ntree
        self.F = F
        self.kNN = kNN
        self.kNNk = kNNk
        self.radius = radius
        C_y = S_data_train.iloc[:, -1]
        C_y = pd.DataFrame(C_y, columns=["class"])
        self.out_of_bag = []
        self.ensemble = self.build_ensemble(S_data_train, C_y)

    def build_ensemble(self, S, C):
        """
        Builds the ensemble of classifiers.

        Args:
        S (pandas DataFrame): Training data.
        C (pandas DataFrame): Labels for the training data.

        Returns:
        list: The ensemble of classifiers.
        """
        ensemble = []
        bootstrap, oob = self.create_bootstrap(S, C)
        self.out_of_bag.append(oob)
        # Train and test algorithms
        ensemble.append(
            random_forest.RandomForest(bootstrap.copy(), self.C_labels.copy(), self.criteria, self.simple, self.WeakLearn, self.ntree+1, self.F, self.kNN, self.kNNk))
        ensemble.append(
            random_forest.RandomForest(bootstrap.copy(), self.C_labels.copy(), self.criteria, self.simple, self.WeakLearn, self.ntree, self.F, self.kNN, self.kNNk))
        # Convert DataFrame to list
        boot_list = bootstrap.copy().values.tolist()
        # Separate the last column
        boot_y = [row[-1] for row in boot_list]
        boot_X = [row[:-1] for row in boot_list]
        ensemble.append(kNN.c_k_NN_algorithm(boot_X, boot_y, self.kNNk[0]))
        ensemble.append(kNN.c_k_NN_algorithm(boot_X, boot_y, self.kNNk[1]))
        ensemble.append(RadiusNeighborsClassifier.RadiusNeighborsClassifier(boot_X, boot_y, self.radius[0]))
        ensemble.append(RadiusNeighborsClassifier.RadiusNeighborsClassifier(boot_X, boot_y, self.radius[1]))
        return ensemble

    def create_bootstrap(self, S, C):
        """
        Creates a bootstrap sample of the training data.

        Args:
        S (pandas DataFrame): Training data.
        C (pandas DataFrame): Labels for the training data.

        Returns:
        tuple: The bootstrap sample and the out-of-bag sample.
        """
        x_train_set, x_test, y_train_set, y_test = train_test_split(S, C, test_size=1 - self.F)

        x_bootstrap, x_unused, y_bootstrap, y_unused = train_test_split(x_train_set, y_train_set,
                                                                            test_size=len(y_train_set) - len(y_test))
        return sklearn.utils.shuffle(x_bootstrap).reset_index(drop=True), \
            sklearn.utils.shuffle(x_test).reset_index(drop=True)

    def test(self, x):
        """
        Tests the ensemble on a test set.

        Args:
        x (pandas DataFrame): Test data.

        Returns:
        list: The predictions of the ensemble for the test set.
        """
        # out of bag is used to select the best trees only
        out_of_bag_copy = self.out_of_bag.copy()
        ensemble_copy = self.ensemble.copy()
        voting = []
        predictions = []
        for ht in range(len(self.ensemble)):
            if ht >= 2:
                x_list = x.values.tolist()
                predictions.append(ensemble_copy[ht].test(x_list))
            else:
                predictions.append(ensemble_copy[ht].test(x))
        for col_index in range(len(predictions[0])):
            column_values = [row[col_index] for row in predictions]
            values_counts = {value: column_values.count(value) for value in set(column_values)}
            mode_value = max(values_counts, key=values_counts.get)
            voting.append(mode_value)
        return voting
