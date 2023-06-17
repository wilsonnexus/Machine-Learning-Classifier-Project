# Author: Wilson Neira
# Evaluating Classifiers
from sklearn.compose import ColumnTransformer
from random_forest import *
from stratified_cross_validation import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
from decision_tree_copy import *


import pandas as pd
import numpy as np


def transform_labels(df):
    for i, column_name in enumerate(df.columns):
        column = df[column_name]
        if pd.api.types.is_numeric_dtype(column):
            df.rename(columns={column_name: str(i)}, inplace=True)
        else:
            df.rename(columns={column_name: column_name}, inplace=True)
    return df


def import_dataset(file_dir):
    file_dataset = pd.read_csv(file_dir, delimiter="\t|,", engine="python")
    # Remove the "#" character from each column label
    file_dataset.columns = [col.replace('#', '') for col in file_dataset.columns]
    # Remove the spacing in column labels
    file_dataset.columns = file_dataset.columns.str.replace(' ', '')
    # Check if the first column label is "class"
    if file_dataset.columns[0] == "class":
        # Remove the first column containing the class labels and append it as the last column
        class_labels = file_dataset.pop(file_dataset.columns[0])
        file_dataset.insert(len(file_dataset.columns), "class", class_labels)
    return file_dataset


def real_ls_copy(list_origin):
    new_list = []
    for val in list_origin:
        if isinstance(val, list):
            new_list.append(real_ls_copy(val))
        else:
            new_list.append(val)
    return new_list


def normalize(x_train, x_test):
    train_copy = real_ls_copy(x_train)
    test_copy = real_ls_copy(x_test)
    num_features = len(train_copy[0])
    max_vals = [float('-inf')] * num_features
    min_vals = [float('inf')] * num_features
    for d in train_copy:
        for i in range(num_features):
            if isinstance(d[i], str):
                continue
            max_vals[i] = max(max_vals[i], float(d[i]))
            min_vals[i] = min(min_vals[i], float(d[i]))

    norm_train = []
    for d in train_copy:
        data_vals = []
        for i in range(num_features):
            if isinstance(d[i], str):
                data_vals.append(d[i])
            else:
                data_vals.append(np.round((float(d[i]) - min_vals[i]) / (max_vals[i] - min_vals[i]), 6))
        norm_train.append(data_vals)

    norm_test = []
    for d in test_copy:
        data_vals = []
        for i in range(num_features):
            if isinstance(d[i], str):
                data_vals.append(d[i])
            else:
                data_vals.append(np.round((float(d[i]) - min_vals[i]) / (max_vals[i] - min_vals[i]), 6))
        norm_test.append(data_vals)
    return norm_train, norm_test


def a_attri_class_split(shuffled_dataset):
    shuffled_dataset = shuffled_dataset.values.tolist()
    x_dataset = real_ls_copy(shuffled_dataset)
    y_dataset = []
    for i in range(0, len(shuffled_dataset)):
        y_dataset.append(x_dataset[i][len(shuffled_dataset[0])-1])
        x_dataset[i].remove(x_dataset[i][len(shuffled_dataset[0])-1])
    return x_dataset, y_dataset


def b_rand_partition(x_dataset, y_dataset):
    x_train_set_80, x_test_set_20, y_train_set_80, y_test_set_20 = train_test_split(x_dataset, y_dataset, test_size=0.2)
    return x_train_set_80, x_test_set_20, y_train_set_80, y_test_set_20


def de_calculate_accuracy(y_data, y_pred_data):
    correct = 0
    for i in range(0, len(y_data)):
        if y_data[i] == y_pred_data[i]:
            correct = correct+1
    return correct/len(y_data)


def SCV_accuracy(dataset, training, normalized, algorithm, simple, ntree, kNN, kNNk, rnc, Ensemble, radius):
    dataset_copy = dataset.copy()
    labels = dataset_copy.columns.tolist()
    labels = labels[0: len(labels) - 1]
    labels = real_ls_copy(labels)
    # Part a
    shuffled_dataset = sklearn.utils.shuffle(dataset_copy)

    # Part b
    x_dataset, y_dataset = a_attri_class_split(shuffled_dataset)
    x_train_set_80, x_test_set_20, y_train_set_80, y_test_set_20 = b_rand_partition(x_dataset, y_dataset)
    x_train_set_80 = x_dataset
    y_train_set_80 = y_dataset

    if normalized:
        x_train_set_80, x_test_set_20 = normalize(x_train_set_80, x_test_set_20)

    # Part c
    x_train_df = pd.DataFrame(x_train_set_80, columns=labels)
    x_test_df = pd.DataFrame(x_test_set_20, columns=labels)
    y_train_df = pd.DataFrame(y_train_set_80, columns=['class'])
    y_test_df = pd.DataFrame(y_test_set_20, columns=['class'])
    data_train_df = pd.concat([x_train_df, y_train_df], axis=1)

    SCV = StratifiedCrossValidation(data_train_df, x_train_df.columns.tolist(), algorithm,
                                    simple, DecisionTree, RandomForest, ntree, 2/3, 10, kNN, kNNk, rnc, Ensemble, radius)
    return SCV.performance


def k_NN_avg_std_accuracies(dataset, training, normalized, algorithm, simple, ntree, kNN, kNNk, rnc, Ensemble, radius):
    dataset_copy = dataset.copy()
    # Q1.2
    accuracies = []
    std_accuracies = []
    f1_scores = []
    std_f1_scores = []
    for k in range(1, 52, 2):
        accuracy, std_accuracy, precision, std_precision, recall, \
            std_recall, f1_score, std_f1_score = SCV_accuracy(dataset, False, True, algorithm, simple, ntree, kNN, k, rnc, Ensemble, radius)
        accuracies.append(accuracy)
        std_accuracies.append(std_accuracy)
        f1_scores.append(f1_score)
        std_f1_scores.append(std_f1_score)

    return accuracies, std_accuracies, f1_scores, std_f1_scores


def RNC_avg_std_accuracies(dataset, training, normalized, algorithm, simple, ntree, kNN, kNNk, rnc, Ensemble, radius):
    dataset_copy = dataset.copy()
    # Q1.2
    accuracies = []
    std_accuracies = []
    f1_scores = []
    std_f1_scores = []
    #for k in range(6, 151, 9):
    for k in range(1, 100, 9):
        accuracy, std_accuracy, precision, std_precision, recall, \
            std_recall, f1_score, std_f1_score = SCV_accuracy(dataset, False, True, algorithm, simple, ntree, kNN, round(k*0.01, 2), rnc, Ensemble, radius)
        accuracies.append(accuracy)
        std_accuracies.append(std_accuracy)
        f1_scores.append(f1_score)
        std_f1_scores.append(std_f1_score)

    return accuracies, std_accuracies, f1_scores, std_f1_scores

def run_SCV(algorithm, simple, file_dir, name, kNN, kNNk, rnc, Ensemble, radius):
    dataset = import_dataset(file_dir)
    if not kNN and not rnc and not Ensemble:
        ntrees = [1, 5, 10, 20, 30, 40, 50]
        accuracies = []
        std_accuracies = []
        precisions = []
        std_precisions = []
        recalls = []
        std_recalls = []
        f1_scores = []
        std_f1_scores = []
        for ntree in ntrees:
            accuracy, std_accuracy, precision, std_precision, recall, \
                std_recall, f1_score, std_f1_score = SCV_accuracy(dataset, False, True, algorithm, simple, ntree, kNN, kNNk, rnc, Ensemble, radius)
            accuracies.append(accuracy)
            std_accuracies.append(std_accuracy)
            precisions.append(precision)
            std_precisions.append(std_precision)
            recalls.append(recall)
            std_recalls.append(std_recall)
            f1_scores.append(f1_score)
            std_f1_scores.append(std_f1_score)


        print("Accuracies:")
        print(accuracies)
        print("F1-Score")
        print(f1_scores)
        # Create a new figure
        fig_accuracy, ax_accuracy = plt.subplots()
        # Create a line plot with error bars
        ax_accuracy.errorbar(ntrees, accuracies, yerr=std_accuracies, fmt='-o', capsize=4, color="black")
        # Set the title and axis labels
        ax_accuracy.set_title(name + ' Average Accuracy By ntree')
        ax_accuracy.set_xlabel('(Value of ntree)')
        ax_accuracy.set_ylabel('(Accuracy)')

        # Set the x-axis tick labels
        ax_accuracy.set_xticks(ntrees)
        ax_accuracy.set_xticklabels(ntrees)


        # Create a new figure
        fig_f1_score, ax_f1_score = plt.subplots()
        # Create a line plot with error bars
        ax_f1_score.errorbar(ntrees, f1_scores, yerr=std_f1_scores, fmt='-o', capsize=4, color="black")
        # Set the title and axis labels
        ax_f1_score.set_title(name + ' Average F1-Score By ntree')
        ax_f1_score.set_xlabel('(Value of ntree)')
        ax_f1_score.set_ylabel('(F1-Score)')

        # Set the x-axis tick labels
        ax_f1_score.set_xticks(ntrees)
        ax_f1_score.set_xticklabels(ntrees)
    if kNN and not rnc and not Ensemble:
        accuracies, std_accuracies, f1_scores, std_f1_scores = k_NN_avg_std_accuracies(dataset, True, True, algorithm, simple, 1, kNN, kNNk, rnc, Ensemble, radius)
        k_vals = [x for x in range(1, 52, 2)]
        print("Accuracies:")
        print(accuracies)
        print("F1-Score")
        print(f1_scores)
        # Create a new figure
        fig_accuracy, ax_accuracy = plt.subplots()
        # Create a line plot with error bars
        ax_accuracy.errorbar(k_vals, accuracies, yerr=std_accuracies, fmt='-o', capsize=4, color="black")
        # Set the title and axis labels
        ax_accuracy.set_title(name + ' Average Accuracy By k Value')
        ax_accuracy.set_xlabel('(Value of k)')
        ax_accuracy.set_ylabel('(Accuracy)')

        # Set the x-axis tick labels
        ax_accuracy.set_xticks(k_vals)
        ax_accuracy.set_xticklabels(k_vals)

        # Create a new figure
        fig_f1_score, ax_f1_score = plt.subplots()
        # Create a line plot with error bars
        ax_f1_score.errorbar(k_vals, f1_scores, yerr=std_f1_scores, fmt='-o', capsize=4, color="black")
        # Set the title and axis labels
        ax_f1_score.set_title(name + ' Average F1-Score By k Value')
        ax_f1_score.set_xlabel('(Value of k)')
        ax_f1_score.set_ylabel('(F1-Score)')

        # Set the x-axis tick labels
        ax_f1_score.set_xticks(k_vals)
        ax_f1_score.set_xticklabels(k_vals)
    if rnc and not Ensemble:
        accuracies, std_accuracies, f1_scores, std_f1_scores = RNC_avg_std_accuracies(dataset, True, True, algorithm,
                                                                                       simple, 1, kNN, kNNk, rnc, Ensemble, radius)
        #k_vals = [round(0.01 * x, 2) for x in range(6, 151, 9)]
        k_vals = [round(0.01 * x, 2) for x in range(1, 100, 9)]
        print("Accuracies:")
        print(accuracies)
        print("F1-Score")
        print(f1_scores)
        # Create a new figure
        fig_accuracy, ax_accuracy = plt.subplots()
        # Create a line plot with error bars
        ax_accuracy.errorbar(k_vals, accuracies, yerr=std_accuracies, fmt='-o', capsize=4, color="black")
        # Set the title and axis labels
        ax_accuracy.set_title(name + ' Average Accuracy By Radius')
        ax_accuracy.set_xlabel('(Value of Radius)')
        ax_accuracy.set_ylabel('(Accuracy)')

        # Set the x-axis tick labels
        ax_accuracy.set_xticks(k_vals)
        ax_accuracy.set_xticklabels(k_vals)

        # Create a new figure
        fig_f1_score, ax_f1_score = plt.subplots()
        # Create a line plot with error bars
        ax_f1_score.errorbar(k_vals, f1_scores, yerr=std_f1_scores, fmt='-o', capsize=4, color="black")
        # Set the title and axis labels
        ax_f1_score.set_title(name + ' Average F1-Score By Radius')
        ax_f1_score.set_xlabel('(Value of Radius)')
        ax_f1_score.set_ylabel('(F1-Score)')

        # Set the x-axis tick labels
        ax_f1_score.set_xticks(k_vals)
        ax_f1_score.set_xticklabels(k_vals)

    if Ensemble:
        ntrees = [1, 5, 10, 20, 30]
        #ntrees = [1]
        accuracies = []
        std_accuracies = []
        precisions = []
        std_precisions = []
        recalls = []
        std_recalls = []
        f1_scores = []
        std_f1_scores = []
        for ntree in ntrees:
            accuracy, std_accuracy, precision, std_precision, recall, \
                std_recall, f1_score, std_f1_score = SCV_accuracy(dataset, False, True, algorithm, simple, ntree, kNN,
                                                                  kNNk, rnc, Ensemble, radius)
            accuracies.append(accuracy)
            std_accuracies.append(std_accuracy)
            precisions.append(precision)
            std_precisions.append(std_precision)
            recalls.append(recall)
            std_recalls.append(std_recall)
            f1_scores.append(f1_score)
            std_f1_scores.append(std_f1_score)
            print("almost:", ntree)

        print("Accuracies:")
        print(accuracies)
        print("F1-Score")
        print(f1_scores)
        # Create a new figure
        fig_accuracy, ax_accuracy = plt.subplots()
        # Create a line plot with error bars
        ax_accuracy.errorbar(ntrees, accuracies, yerr=std_accuracies, fmt='-o', capsize=4, color="black")
        # Set the title and axis labels
        ax_accuracy.set_title(name + ' Average Ensemble Accuracy By ntree')
        ax_accuracy.set_xlabel('(Value of ntree)')
        ax_accuracy.set_ylabel('(Accuracy)')

        # Set the x-axis tick labels
        ax_accuracy.set_xticks(ntrees)
        ax_accuracy.set_xticklabels(ntrees)

        # Create a new figure
        fig_f1_score, ax_f1_score = plt.subplots()
        # Create a line plot with error bars
        ax_f1_score.errorbar(ntrees, f1_scores, yerr=std_f1_scores, fmt='-o', capsize=4, color="black")
        # Set the title and axis labels
        ax_f1_score.set_title(name + ' Average Ensemble F1-Score By ntree')
        ax_f1_score.set_xlabel('(Value of ntree)')
        ax_f1_score.set_ylabel('(F1-Score)')

        # Set the x-axis tick labels
        ax_f1_score.set_xticks(ntrees)
        ax_f1_score.set_xticklabels(ntrees)

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    # Set options to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Uncomment the code you want to run and please adjust the hyperparameters according to our responses
    # run_SCV(impurity criterion, simple heuristic, data file, data name, kNN?, k for KNN, RNC?, Ensemble, Radius)
    # Please only put a list in k for KNN and radius for Ensemble

    # Random Forrest
    run_SCV("InfoGain", "yes", "diabetes.csv", "Diabetes", False, 1, False, False, 2)
    run_SCV("InfoGain", "yes", "contraceptive_method.csv", "Contraceptive", False, 1, False, False, 2)

    # kNN
    #run_SCV("InfoGain", "yes", "diabetes.csv", "Diabetes", True, 1, False, False, 2)
    #run_SCV("InfoGain", "yes", "contraceptive_method.csv", "Contraceptive", True, 1, False, False, 2)

    # RadiusNeighborsClassifier
    # run_SCV("InfoGain", "yes", "diabetes.csv", "Diabetes", True, 1, False, False, 2)
    # run_SCV("InfoGain", "yes", "contraceptive_method.csv", "Contraceptive", True, 1, False, False, 2)

    # Ensemble
    #run_SCV("InfoGain", "yes", "diabetes.csv", "Diabetes", False, [1, 2], False, True, [1.5, 1.41])
    # run_SCV("InfoGain", "yes", "contraceptive_method.csv", "Contraceptive", False, [1, 2], False, True, [1.5, 1.41])



    # Show the plot
    plt.show()