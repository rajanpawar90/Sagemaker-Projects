# -*- coding: utf-8 -*-

# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function

import os
import json
import pickle 
import sys
import traceback 

import pandas as pd
from sklearn import tree 

# These are the paths to where SageMaker mounts interesting things in your container.
prefix = '/opt/ml/'

input_path = os.path.join(prefix,'input','data')
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
channel_name = 'training'
training_path = os.path.join(input_path, channel_name)

# The function to execute the training.
def train():
    print('Starting The Training')
    try:
        #Read hyperparameters as json object
        with open(param_path, 'r') as tc:
            trainingParams = json.load(tc)
        
        #Take the set of files and read them all into a single pandas dataframe
        input_files=[os.path.join(training_path, file) for file in os.listdir(training_path)]
        if len(input_files) ==0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(training_path, channel_name))
        raw_data = [pd.read_csv(file, header=None) for file in input_files]
        train_data = pd.concat(raw_data)
    
        # labels are in the first column
        train_y = train_data.ix[:,0]
        train_X = train_data.ix[:,1:]
        
        #use hyperparameter from the json file
        max_leaf_nodes = trainingParams.get('max_leaf_nodes', None)
        if max_leaf_nodes is not None:
            max_leaf_nodes = int(max_leaf_nodes)
        
        # Now use scikit-learn's decision tree classifier to train the model.
        decision_tree_classifier = tree.DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes)
        decision_tree_classifier.fit(train_X, train_y)
        
        #save the model
        with open(os.path.join(model_path, 'decision-tree-model.pkl'), 'w') as out:
            pickle.dump(decision_tree_classifier, out)
        print('Training Complete')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: '+ str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: '+str(e) + '\n' + trc, file=sys.stderr)
        sys.exit(255)
    
if __name__ == '__main__':
    train()
    
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
