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
#from ML_Utils.ML_Utils import *
from sklearn.model_selection import train_test_split 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #hyperparameter intake
    parser.add_argument('--max_leaf_nodes', type=int, default=-1)
    
    #sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    
    # Take the set of files from train directory and read them all into a single pandas dataframe
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    raw_data = [pd.read_csv(file, header=None, engine='python') for file in input_files]
    train_data = pd.concat(raw_data)
    
    # labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]
    # as your training my require in the ArgumentParser above.
    max_leaf_nodes = args.max_leaf_nodes
    
    #pass max leaf node accepted from arg parse to decision tree as hyperparameter
    tree_classifier = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    tree_classifier.fit(train_X, train_y)
    
    #print the coefficients of trained classifier and save them
    joblib.dump(tree_classifier, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    #Deserialized and return fitted model
    #Note that this should have the same name as the serialized model in the main method
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf