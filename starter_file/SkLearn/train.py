# Modified from https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/

import argparse
import os

# importing necessary libraries
import numpy as np
import pandas as pd

from sklearn import datasets
#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from azureml.core.workspace import Workspace
#from azureml.train.automl import AutoMLConfig
from azureml.core.dataset import Dataset

import joblib

from azureml.core.run import Run

run = Run.get_context()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel', type=str, default='linear',
                        help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0,
                        help='Penalty parameter of the error term')

    args = parser.parse_args()
    run.log('Kernel type', np.str(args.kernel))
    run.log('Penalty', np.float(args.penalty))

    # loading the iris dataset
    #iris = datasets.load_iris()

    # X -> features, y -> label
    #X = iris.data
    #y = iris.target

    ws = Workspace.from_config()

    key = "Heart-Failure"

    if key in ws.datasets.keys(): 
        dataset = ws.datasets[key]
    
    df = dataset.to_pandas_dataframe()

    y = df[df.columns[-1]]
    X = df.drop(df.columns[-1],axis=1)

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # training a linear SVM classifier
    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel=args.kernel, C=args.penalty).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)

    # model accuracy for X_test
    #accuracy = svm_model_linear.score(X_test, y_test)
    #print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    #run.log('Accuracy', np.float(accuracy))
    # creating a confusion matrix
    #cm = confusion_matrix(y_test, svm_predictions)
    #print(cm)

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(svm_model_linear, 'outputs/model.joblib')


if __name__ == '__main__':
    main()