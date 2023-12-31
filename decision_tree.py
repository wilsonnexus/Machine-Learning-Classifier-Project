# Author: Wilson Neira

import pandas as pd
import numpy as np


class DecisionTree:
    """
    A class for implementing a Decision Tree algorithm.

    Args:
    data_train (list): List of training data.
    labels (list): List of labels for the features.
    algorithm (str): The type of algorithm to use for splitting (either "InfoGain" or "Gini").
    simple (bool): Indicator to decide if it's a simple decision tree.
    RF (bool): Indicator to decide if it's used in Random Forests.

    Attributes:
    data_train (list): List of training data.
    labels (list): List of labels for the features.
    algorithm (str): The type of algorithm to use for splitting (either "InfoGain" or "Gini").
    simple (bool): Indicator to decide if it's a simple decision tree.
    RF (bool): Indicator to decide if it's used in Random Forests.
    """
    class Node:
        """
        A class for the node in a decision tree.

        Args:
        label (str): The label of the node.
        node_type (str): Type of the node ("decision" or "leaf").

        Attributes:
        label (str): The label of the node.
        node_type (str): Type of the node ("decision" or "leaf").
        """
        node_type = ""
        label = None
        testlabel = ""
        edge = {}
        majority = -1
        threshold = -1
        depth = 0
        parent = None
        def __init__(self, label, node_type):
            self.label = label
            self.node_type = node_type

        def curr_depth(self):
            """
            Get the depth of the current node in the decision tree.

            Returns:
            The depth of the current node in the decision tree.
            """
            node = self
            while node.parent is not None:
                self.depth += 1
                node = node.parent
            return self.depth

    def __init__(self, data_train, labels, algorithm, simple, RF = True):
        """
        Initialize a DecisionTree.
        """
        self.labels = labels
        self.RF = RF
        self.algorithm = algorithm
        self.simple = simple
        y = data_train.iloc[:, -1]
        y = pd.DataFrame(y, columns=["class"])
        self.root = self.build_tree(data_train, y)

    def build_tree(self, X, y):
        """
        Build the decision tree.

        Parameters:
        X : The feature data.
        y : The labels.

        Returns:
        The root node of the built decision tree.
        """
        X_copy = X.copy()
        # m ≈ √X mechanism to select m attributes at random from the complete set of attributes
        if self.RF and len(self.labels) != 0:
            len_X_sqr = int(np.round(np.sqrt(X_copy.shape[1])))
            selected_columns = np.random.choice(X_copy.columns[:-1], size=len_X_sqr, replace=False)
            self.labels = selected_columns.tolist()
            # Concatenate the selected columns with the last column
            X_copy = pd.concat([X_copy[selected_columns], X_copy['class']], axis=1)

        y = X.iloc[:, -1]

        y = pd.DataFrame(y, columns=["class"])
        node = self.Node(label=-1, node_type="decision")
        # update labels
        labels_copy = self.labels
        # Base cases: unique class or no labels left
        node.majority = y['class'].mode()[0]

        if len(labels_copy) == 0:
            node.node_type = "leaf"
            node.label = y['class'].mode()[0]
            return node

        if self.simple.lower() == "yes":
            max_counted = y["class"].value_counts()
            percent = max_counted.iloc[0] / len(y)
            if percent >= 0.85:
                node.node_type = "leaf"
                node.label = y['class'].mode()[0]
                return node
        else:
            if len(np.unique(y)) == 1:
                node.node_type = "leaf"
                node.label = (y.iloc[0]).item()
                return node

        # minimal size for split criterion
        if len(X_copy) <= 10:
            node.node_type = "leaf"
            node.label = y['class'].mode()[0]
            return node

        # Calculate information gain for each label
        best_attr = float('-inf')
        max_A = None
        threshold = float('-inf')
        i = 0
        if self.algorithm == "InfoGain":
            for A in labels_copy:
                info_gain, threshold_curr = self.information_gain(X, A)
                if info_gain > best_attr:
                    threshold = threshold_curr
                    best_attr = info_gain
                    max_A = A
                i = i + 1
        else:
            # Gini Criterion
            best_attr = float('+inf')
            for A in labels_copy:
                gini, threshold_curr = self.gini_gain(X, A)
                if gini < best_attr:
                    threshold = threshold_curr
                    best_attr = gini
                    max_A = A
                i = i + 1
        node.testlabel = max_A
        node.threshold = threshold

        # minimal gain criterion
        if best_attr < 0.01:
            node.node_type = "leaf"
            node.label = y['class'].mode()[0]
            return node

        sub_data = []
        Dv_df = []
        if max_A.isdigit():
            X_copy_sorted = (X_copy.sort_values(max_A).reset_index(drop=True)).copy()
            split = 0
            for v in X_copy_sorted[max_A]:
                if v > threshold:
                    break
                else:
                    split +=1
            y = X_copy_sorted.iloc[:, -1]
            y = pd.DataFrame(y, columns=["class"])
            Dv_left = X_copy_sorted.iloc[:split, :]
            Dv_right = X_copy_sorted.iloc[split:, :]
            Dv_left = pd.concat([Dv_left], axis=1)
            Dv_right = pd.concat([Dv_right], axis=1)
            Dv_df.append(Dv_left)
            Dv_df.append(Dv_right)
            for Dv in Dv_df:
                Dv = Dv.drop(max_A, axis=1)
                sub_data.append(Dv)
        else:
            V = (sorted(X_copy[max_A].unique())).copy()
            for v in V:
                Dv = (X.loc[X_copy[max_A] == v]).copy()
                Dv = Dv.drop(max_A, axis=1)
                sub_data.append(Dv)


        labels_copy.remove(max_A)

        edge = {}
        i_data = 0
        for sub_v in sub_data:
            if sub_v.size == 0:
                node.node_type = "leaf"
                node.label = node.majority
                node.threshold = threshold
                return node
            if node.curr_depth() + 1 > 10:
                node.node_type = "leaf"
                node.label = y['class'].mode()[0]
                node.threshold = threshold
                return node

            y = sub_v.iloc[:, -1]
            y = pd.DataFrame(y, columns=["class"])
            T = self.build_tree(sub_v, y)
            T.parent = node
            if max_A.isdigit():
                A_val = "<=" if i_data == 0 else ">"
            else:
                A_val = V[i_data]
            edge[A_val] = T
            i_data = i_data + 1
        node.edge = edge
        return node

    def entropy(self, data):
        """
        Calculate the entropy of the given data.

        Parameters:
        data : The data to calculate entropy.

        Returns:
        The entropy of the data.
        """
        data_copy = data.copy()
        total_entropy = 0
        n = len(data_copy)
        if n == 0:
            return 0
        V = (data_copy['class'].unique()).copy()
        for v in V:
            total = len(data_copy[data_copy['class'] == v])
            if total == 0:
                return 0
            percentage = total/n
            total_entropy -=  percentage * np.log2(percentage)

        return total_entropy

    def information_gain(self, x_train, label):
        """
        Calculate the information gain and the best threshold for splitting data based on a given attribute.

        Parameters:
        x_train : The training data.
        label : The attribute label.

        Returns:
        The information gain and the best threshold.
        """
        threshold_curr = -1
        x_train_copy = x_train.copy()
        x_train_entropy = self.entropy(x_train_copy)
        is_numeric = pd.api.types.is_numeric_dtype(x_train_copy[label])
        if is_numeric:
            label_data = x_train_copy[[label, 'class']].sort_values(label).reset_index(drop=True)
        else:
            label_data = x_train_copy[[label, 'class']].sort_values(label, key=lambda col: col.astype(str)).reset_index(
                drop=True)
        n = label_data.shape[0]
        min_entropy = float('inf')
        best_threshold= None
        if label.isdigit():
            thresholds = []
            unique_vals = label_data[label].unique()
            sorted_vals = np.sort(unique_vals)
            for i in range(len(sorted_vals) - 1):
                thresholds.append((sorted_vals[i] + sorted_vals[i + 1]) / 2)

            for threshold_curr in thresholds:
                decisions = [label_data[label_data[label] <= threshold_curr],
                             label_data[label_data[label] > threshold_curr]]
                avg_entropy = 0 #Delete this if the code does not work
                for decision in decisions:
                    decision_entropy = self.entropy(decision)
                    avg_entropy += (decision.shape[0] / n) * decision_entropy
                if avg_entropy < min_entropy:
                    min_entropy = avg_entropy
                    best_threshold = threshold_curr
        else:
            V = label_data[label].unique()
            for v in V:
                decision = label_data.loc[label_data[label] == v]
                decision_entropy = self.entropy(decision)
                avg_entropy = (decision.shape[0] / n) * decision_entropy
                if avg_entropy < min_entropy:
                    min_entropy = avg_entropy
                    best_threshold = None
        return (x_train_entropy - min_entropy), best_threshold

    def gini(self, data):
        """
        Calculate the Gini Index of the given data.

        Parameters:
        data : The data to calculate the Gini Index.

        Returns:
        The Gini Index of the data.
        """
        data_copy = data.copy()
        total_gini = 0
        n = len(data_copy)
        if n == 0:
            return 0
        V = data_copy['class'].unique()
        for v in V:
            total = len(data_copy[data_copy['class'] == v])
            if total == 0:
                return 0
            percentage = total / n
            total_gini += percentage * (1 - percentage)
        return total_gini

    def gini_gain(self, x_train, label):
        """
        Calculate the Gini Gain and the best threshold for splitting data based on a given attribute.

        Parameters:
        x_train : The training data.
        label : The attribute label.

        Returns:
        The Gini Gain and the best threshold.
        """
        x_train_copy = x_train.copy()
        x_train_gini = self.gini(x_train_copy)
        label_data = x_train_copy[[label, 'class']].sort_values(label).reset_index(drop=True)
        n = label_data.shape[0]
        min_gini = float('inf')
        best_threshold = None

        if label.isdigit():
            thresholds = []
            unique_vals = label_data[label].unique()
            sorted_vals = np.sort(unique_vals)
            for i in range(len(sorted_vals) - 1):
                thresholds.append((sorted_vals[i] + sorted_vals[i + 1]) / 2)

            for threshold_curr in thresholds:
                decisions = [label_data[label_data[label] <= threshold_curr],
                             label_data[label_data[label] > threshold_curr]]
                avg_gini = 0
                for decision in decisions:
                    decision_gini = self.gini(decision)
                    avg_gini += (decision.shape[0] / n) * decision_gini
                if avg_gini < min_gini:
                    min_gini = avg_gini
                    best_threshold = threshold_curr
        else:
            V = label_data[label].unique()
            for v in V:
                decision = label_data.loc[label_data[label] == v]
                decision_gini = self.gini(decision)
                avg_gini = (decision.shape[0] / n) * decision_gini
                if avg_gini < min_gini:
                    min_gini = avg_gini
                    best_threshold = None
        return x_train_gini - min_gini, best_threshold

    def predict(self, DT, X_test):
        """
        Predict the class for a given test instance.

        Parameters:
        DT : The decision tree to use for prediction.
        X_test : The test instance.

        Returns:
        The predicted class for the test instance.
        """
        predict = DT.majority
        if DT.node_type == 'leaf':
            predict = DT.label
            return predict
        if DT.testlabel.isdigit():
            if X_test.loc[DT.testlabel]<= DT.threshold:
                nextDT = DT.edge['<=']
            else:
                nextDT = DT.edge['>']
        else:
            if X_test.loc[DT.testlabel] not in DT.edge:
                return predict
            nextDT = DT.edge[X_test.loc[DT.testlabel]]
        return self.predict(nextDT, X_test)

    def test(self, data, DT):
        """
        Test the decision tree on the given dataset.

        Parameters:
        data : The dataset to test.
        DT : The decision tree to use for testing.

        Returns:
        The predicted classes for each instance in the dataset.
        """
        predictions = []
        for index, ins in data.iterrows():
            predictions.append(self.predict(DT, ins))
        return predictions







