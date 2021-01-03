# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 10:40:26 2020

@author: Rajan
"""
#https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_iris/scikit_learn_iris.py
from __future__ import print_function
import argparse
import joblib
import pandas as pd
import os
import sklearn
from sklearn import tree, ensemble, linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, \
    roc_auc_score, roc_curve, recall_score, precision_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #hyperparameter intake
    #parser.add_argument('--max_leaf_nodes', type=int, default=-1)
    
    #sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    args = parser.parse_args()
    
    #Get train and test directories from arg parser
    train_data_dir = args.train
    test_data_dir = args.test
    
    # Take the set of files from train directory and read them all into a single pandas dataframe
    # convert code below to reac pickle
    train_X = pd.read_csv([os.path.join(train_data_dir, file) for file in os.listdir(train_data_dir) if "train_X" in file][0], engine='python')
    train_y = pd.read_csv([os.path.join(train_data_dir, file) for file in os.listdir(train_data_dir) if "train_y" in file][0], engine='python')
    test_X = pd.read_csv([os.path.join(test_data_dir, file) for file in os.listdir(test_data_dir) if "test_X" in file][0], engine='python')
    test_y = pd.read_csv([os.path.join(test_data_dir, file) for file in os.listdir(test_data_dir) if "test_X" in file][0], engine='python')

    #tree classifier 
    #max_leaf_nodes = 30 
    #tree_classifier = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    #tree_classifier.fit(train_X, train_y)
    # print the coefficients of trained classifier and save them
    #joblib.dump(tree_classifier, os.path.join(args.model_dir, "model.joblib"))
    
    #Create classifiers
    log_reg = LogisticRegression()
    #decision_tree_class = DecisionTreeClassifier()
    #random_for_class = RandomForestClassifier()
    #svm_class = SVC()
    #sgd_class = SGDClassifier()

    classifiers = []
    classifiers.append(log_reg)
    #classifiers.append(decision_tree_class)
    #classifiers.append(random_for_class)
    #classifiers.append(svm_class)
    #classifiers.append(sgd_class)

    #for Binary classification  
    #print("Accuracy, F1, Precision, Recall, Precision-Recall curve, roc auc score, auc curve ")
    for classifier in classifiers:
        classifier.fit(train_X, train_y)
        preds = classifier.predict(test_X)
        joblib.dump(classifier, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    #Deserialized and return fitted model
    #Note that this should have the same name as the serialized model in the main method
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
# def get_classification_metrics(y_actuals, y_preds, binary):
#     if binary:
#         return accuracy_score(y_actuals, y_preds), \
#             f1_score(y_actuals, y_preds), \
#                 precision_score(y_actuals, y_preds), \
#                     recall_score(y_actuals, y_preds), \
#                         precision_recall_curve(y_actuals, y_preds), \
#                             roc_auc_score(y_actuals, y_preds), \
#                                 roc_curve(y_actuals, y_preds)                                 
#     else:
#         return accuracy_score(y_actuals, y_preds)