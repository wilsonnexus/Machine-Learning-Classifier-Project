Machine Learning Modified Final
Once hit run, the program will run all the homework probelms, including all the extra credit ones. For verifying the code, it could be recommended to to commont some run_SCV after if __name__ == '__main__':
Set up the hyper-parameters you want.
There is Random Forests, kNN, RadiusNeighborsClassifier, and Ensemble of the previous mentioned algorithms.
For all datasets I used stratified cross validation. 

For the random forrest I implemented in addition to the original, to work with numrical and categorical attributes, the gini criteria, minimal size for split criterion where size is 10, minimal gain criterion where gain is 0.01, and maximal depth stopping criterion where depth is 10. 