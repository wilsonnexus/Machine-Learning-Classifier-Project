# Author: Wilson Neira

import pandas as pd
import numpy as np
import ensemble
from kNN import *
from RadiusNeighborsClassifier import *


class StratifiedCrossValidation:
    """
    A class for implementing the Stratified Cross Validation algorithm.

    Args:
    s_data_train (pandas.DataFrame): The training data.
    c_labels (list): The labels.
    criteria (object): The criteria to be used for decision making.
    simple (bool): Specifies if the model should be a simple or complex one.
    weak_learn (object): The weak learning model.
    bagging (object): The bagging model.
    ntree (int): Number of trees for the ensemble learning algorithm.
    F (int): Specifies the number of features to consider when looking for the best split.
    k (int): Number of folds in the cross validation.
    kNN (bool): Specifies if the k-Nearest Neighbors model should be used.
    kNNk (int): Specifies the number of neighbors to include in the majority voting process for k-NN.
    RNC (bool): Specifies if the Radius Neighbors Classifier model should be used.
    Ensemble (bool): Specifies if the Ensemble model should be used.
    radius (float): The range in which to search for the neighbors for RNC.

    Attributes:
    performance (tuple): The performance metrics of the model. Includes average accuracy, standard deviation of accuracy,
                         average precision, standard deviation of precision, average recall, standard deviation of recall,
                         average f1_score, standard deviation of f1_score.
    """
    def __init__(self, s_data_train, c_labels, criteria, simple, weak_learn, bagging, ntree, F, k, kNN, kNNk, RNC, Ensemble, radius):
        self.c_labels = c_labels
        self.criteria = criteria
        self.simple = simple
        self.weak_learn = weak_learn
        self.bagging = bagging
        self.ntree = ntree
        self.F = F
        self.k = k
        self.kNNk = kNNk
        self.kNN = kNN
        self.RNC = RNC
        self.radius = radius
        self.Ensemble = Ensemble
        C_y = s_data_train.iloc[:, -1]
        C_y = pd.DataFrame(C_y, columns=["class"])
        self.performance = self.cross_validation(s_data_train, C_y)

    def cross_validation(self, S, C):
        """
        Executes the Stratified Cross Validation.

        Args:
        S (pandas.DataFrame): The shuffled DataFrame.
        C (pandas.DataFrame): The labels.

        Returns:
        tuple: Average and standard deviation of accuracy, precision, recall and f1_score.
        """
        S_copy = S.copy()
        # Shuffle the DataFrame randomly
        S_shuffled = S.sample(frac=1, random_state=42)

        # Split the DataFrame into 5 equal-sized subsets
        folds = self.create_folds(S_shuffled, self.k)
        # models_performance = []
        models_accuracy = []
        models_precision = []
        models_recall = []
        models_f1_score = []
        for i_test in range(len(folds)):
            folds_train = folds[:]
            fold_test = folds_train.pop(i_test)
            fold_test_class = fold_test['class'].tolist()
            fold_test = fold_test.drop('class', axis=1)
            folds_train = pd.concat(folds_train)
            if self.kNN:
                # Train and test k-Nearest Neighbors
                # Convert DataFrame to list
                folds_list = folds_train.values.tolist()
                # Separate the last column
                folds_y = [row[-1] for row in folds_list]
                folds_X = [row[:-1] for row in folds_list]

                RF = c_k_NN_algorithm(folds_X, folds_y, self.kNNk)
                test_list = fold_test.values.tolist()
                predictions = RF.test(test_list)
            elif self.RNC:
                # Train and test Radius Neighbors Classifier
                # Convert DataFrame to list
                folds_list = folds_train.values.tolist()
                # Separate the last column
                folds_y = [row[-1] for row in folds_list]
                folds_X = [row[:-1] for row in folds_list]
                RF = RadiusNeighborsClassifier(folds_X, folds_y, self.kNNk)
                test_list = fold_test.values.tolist()
                predictions = RF.test(test_list)
            elif self.Ensemble:
                # Train and test multi-algorithm Ensembler
                RF = ensemble.Ensemble(folds_train, self.c_labels, self.criteria, self.simple, self.weak_learn, self.ntree,
                                  self.F, self.kNN, self.kNNk, self.radius)
                predictions = RF.test(fold_test)
            else:
                # Train and test Random Forests
                RF = self.bagging(folds_train, self.c_labels, self.criteria, self.simple, self.weak_learn, self.ntree, self.F, self.kNN, self.kNNk)
                predictions = RF.test(fold_test)
            accuracy_macro, precision_macro, recall_macro, f1_score_macro = self.model_performance(fold_test_class, predictions)
            models_accuracy.append(accuracy_macro)
            models_precision.append(precision_macro)
            models_recall.append(recall_macro)
            models_f1_score.append(f1_score_macro)

        return sum(models_accuracy)/len(models_accuracy), np.std(models_accuracy), \
            sum(models_precision) / len(models_precision), np.std(models_precision), \
            sum(models_recall)/len(models_recall), np.std(models_recall), \
            sum(models_f1_score)/len(models_f1_score), np.std(models_f1_score)

    def create_folds(self, S, C):
        """
        Creates k folds from the DataFrame S.

        Args:
        S (pandas.DataFrame): The shuffled DataFrame.
        C (pandas.DataFrame): The labels.

        Returns:
        list: List of folds, each fold being a DataFrame.
        """
        # Compute the class distribution
        class_dist = S['class'].value_counts(normalize=True)
        # Compute the number of samples per fold for each class
        samples_per_fold = (class_dist * len(S) / self.k).round().astype(int)
        folds = []
        for i in range(self.k):
            fold_data = []
            # Loop over each class
            for class_name in class_dist.index:
                # Select the rows for this class
                class_data = S[S['class'] == class_name]
                class_data = class_data.sample(frac=1, random_state=i)
                start = i * samples_per_fold[class_name]
                end = (i + 1) * samples_per_fold[class_name]
                fold_class_data = class_data.iloc[start:end]
                fold_data.append(fold_class_data)
            # Concatenate the data for all classes and append it to the folds list
            folds.append(pd.concat(fold_data))
        return folds

    def model_performance(self, true_classes, predictions):
        """
        Computes the performance metrics (accuracy, precision, recall, f1_score) for the model.

        Args:
        true_classes (list): The true labels.
        predictions (list): The predicted labels.

        Returns:
        tuple: Average of accuracy, precision, recall and f1_score.
        """
        unique_classes = sorted(set(true_classes))
        num_classes = len(unique_classes)

        tp_classes = [0] * num_classes
        fp_classes = [0] * num_classes
        tn_classes = [0] * num_classes
        fn_classes = [0] * num_classes

        for i in range(num_classes):
            for j in range(len(true_classes)):
                if true_classes[j] == unique_classes[i] and predictions[j] == unique_classes[i]:
                    tp_classes[i] += 1
                elif true_classes[j] != unique_classes[i] and predictions[j] == unique_classes[i]:
                    fp_classes[i] += 1
                elif predictions[j] == true_classes[j] and predictions[j] != unique_classes[i]:
                    tn_classes[i] += 1
                elif true_classes[j] == unique_classes[i] and predictions[j] != unique_classes[i]:
                    fn_classes[i] += 1

        accuracy_classes = []
        precision_classes = []
        recall_classes = []
        f1_score_classes = []

        for i in range(num_classes):
            accuracy = (tp_classes[i] + tn_classes[i]) / (tp_classes[i] + fp_classes[i] + tn_classes[i] + fn_classes[i])
            # Check if summation is greater than 0 else this result is 0
            precision = tp_classes[i] / (tp_classes[i] + fp_classes[i]) if tp_classes[i] + fp_classes[i] > 0 else 0
            recall = tp_classes[i] / (tp_classes[i] + fn_classes[i]) if tp_classes[i] + fn_classes[i] > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            accuracy_classes.append(accuracy)
            precision_classes.append(precision)
            recall_classes.append(recall)
            f1_score_classes.append(f1_score)
        accuracy_macro = sum(accuracy_classes) / num_classes
        precision_macro = sum(precision_classes) / num_classes
        recall_macro = sum(recall_classes) / num_classes
        f1_score_macro = sum(f1_score_classes) / num_classes
        return accuracy_macro, precision_macro, recall_macro, f1_score_macro
