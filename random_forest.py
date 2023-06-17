# Author: Wilson Neira
# Evaluating the Decision Tree Algorithm
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import decision_tree_copy
""" Random Forest (RF) Algorithm Begins Here"""


class RandomForest:

    def __init__(self, S_data_train, C_labels, criteria, simple, WeakLearn, ntree, F, kNN, kNNk):
        self.C_labels = C_labels
        self.criteria = criteria
        self.simple = simple
        self.WeakLearn = WeakLearn
        self.ntree = ntree
        self.F = F
        self.kNN = kNN
        self.kNNk = kNNk
        C_y = S_data_train.iloc[:, -1]
        C_y = pd.DataFrame(C_y, columns=["class"])
        self.out_of_bag = []
        self.ensemble = self.build_forest(S_data_train, C_y)

    def build_forest(self, S, C):
        ensemble = []
        for t in range(self.ntree):
            bootstrap, oob = self.create_bootstrap(S, C)
            self.out_of_bag.append(oob)
            if self.kNN:
                # Treain and test kNN
                # Convert DataFrame to list
                boot_list = bootstrap.copy().values.tolist()
                # Separate the last column
                boot_y = [row[-1] for row in boot_list]
                boot_X = [row[:-1] for row in boot_list]
                ensemble.append(self.WeakLearn(boot_X, boot_y, self.kNNk))
            else:
                ensemble.append(decision_tree_copy.DecisionTree(bootstrap.copy(), self.C_labels.copy(), self.criteria, self.simple))

        return ensemble

    def create_bootstrap(self, S, C):
        x_train_set, x_test, y_train_set, y_test = train_test_split(S, C, test_size=1 - self.F)

        x_bootstrap, x_unused, y_bootstrap, y_unused = train_test_split(x_train_set, y_train_set,
                                                                            test_size=len(y_train_set) - len(y_test))
        return sklearn.utils.shuffle(x_bootstrap).reset_index(drop=True), \
            sklearn.utils.shuffle(x_test).reset_index(drop=True)

    def test(self, x):
        # out of bag is used to select the best trees only
        out_of_bag_copy = self.out_of_bag.copy()
        ensemble_copy = self.ensemble.copy()
        voting = []
        predictions = []
        for ht in range(len(self.ensemble)):
            if self.kNN:
                x_list = x.values.tolist()
                predictions.append(ensemble_copy[ht].test(x_list))
            else:
                predictions.append(ensemble_copy[ht].test(x, ensemble_copy[ht].root))
        for col_index in range(len(predictions[0])):
            column_values = [row[col_index] for row in predictions]
            values_counts = {value: column_values.count(value) for value in set(column_values)}
            mode_value = max(values_counts, key=values_counts.get)
            voting.append(mode_value)
        return voting
