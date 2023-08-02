# Machine Learning Modified Final
This is a machine learning program implemented in Python that demonstrates classification algorithms on two datasets - diabetes diagnosis and contraceptive method choice. The main algorithms included are:

* Random Forest
* k-Nearest Neighbors (kNN)
* Radius Neighbors Classifier
* Ensemble Classifier
The program utilizes stratified k-fold cross validation to evaluate the models.

## Files
* main.py - The driver code that executes the pipelines
* stratified_cross_validation.py - Implements stratified cross validation
* random_forest.py - Random Forest classifier
* decision_tree_copy.py - Decision Tree used in Random Forest
* kNN.py - kNN classifier
* RadiusNeighborsClassifier.py - Radius Neighbors Classifier
* ensemble.py - Ensemble Classifier
## Algorithms
Random Forest is implemented with additional features like gini split criterion, minimum split size of 10, minimum gain of 0.01, and max depth of 10.

Ensemble Classifier combines Random Forest, kNN, and Radius Neighbors Classifier.

## Datasets
* diabetes.csv - Pima Indians Diabetes data
* contraceptive_method.csv - Contraceptive method choice data
The CSV files have headers and the last column is the class label.

## Usage
To run:

1. Install requirements with pip install -r requirements.txt
2. Run python main.py
main.py executes the pipelines and prints results. Hyperparameters are configured in main.py.

## Extensions
Possible extensions:

* Add more datasets
* Implement additional classifiers
* Use grid search for tuning hyperparameters
* Add metrics like ROC curve
* Containerize with Docker
## Credits
Author: Wilson Neira

This program serves as a course project to demonstrate machine learning skills. The data files are from open repositories.












Once hit run, the program will run all the homework probelms, including all the extra credit ones. For verifying the code, it could be recommended to to commont some run_SCV after if __name__ == '__main__':
Set up the hyper-parameters you want.
There is Random Forests, kNN, RadiusNeighborsClassifier, and Ensemble of the previous mentioned algorithms.
For all datasets I used stratified cross validation. 

For the random forest I implemented in addition to the original, to work with numrical and categorical attributes, the gini criteria, minimal size for split criterion where size is 10, minimal gain criterion where gain is 0.01, and maximal depth stopping criterion where depth is 10. 

Dataset originally from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download
