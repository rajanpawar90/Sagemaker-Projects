# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#sample notebooks reviewed
# https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_randomforest/Sklearn_on_SageMaker_end2end.ipynb 
# https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_iris/scikit_learn_estimator_example_with_batch_transform.ipynb
# https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_inference_pipeline/Inference%20Pipeline%20with%20Scikit-learn%20and%20Linear%20Learner.ipynb


import os
#import ML_Utils.ML_Utils
from ML_Utils.ML_Utils import *
import sklearn
from sklearn.datasets import california_housing, \
    load_breast_cancer, load_wine, load_iris
import sagemaker 
import boto3
import re
from sagemaker import image_uris
from sagemaker import TrainingInput
import pandas as pd
import numpy as np

region = boto3.Session().region_name
bucket= 'sklearn-sagemaker-data'
prefix_wine = 'wine'
prefix_cali = 'california_housing'
prefix_breast_cancer = 'breast_cancer'
prefix_iris = 'iris'
prefix_output = 'output'

sagemaker_session = sagemaker.Session()
iam = boto3.client('iam')
SageMakerRole = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20200112T192472')['Role']['Arn']

s3client = boto3.client('s3')

#Prepare training data in S3

##Iris data in local folder
iris_data = sklearn.datasets.load_iris()
joined_iris = np.insert(iris_data.data, 0, iris_data.target, axis=1) #insert targets before index 0 along axis=1
os.makedirs('AWS_Sagemaker/sklearn_sagemaker_deploy/data', exist_ok=True)
np.savetxt('AWS_Sagemaker/sklearn_sagemaker_deploy/data/iris_data.csv', joined_iris, delimiter=',', fmt='%1.1f, %1.3f, %1.3f, %1.3f, %1.3f')

local_breast_cancer_dir= 'AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer'
local_iris_dir = 'AWS_Sagemaker/sklearn_sagemaker_deploy/data/iris'

#copy contents of local directory into s3 bucket using sagemaker_session.upload_data
train_input_iris = sagemaker_session.upload_data(path= local_iris_dir,
                                            bucket='sklearn-sagemaker-data',
                                            key_prefix='iris')

#Download and save breast cancer data to local directory
breast_cancer_data = sklearn.datasets.load_breast_cancer()
pd.DataFrame(breast_cancer_data.get('data')).to_csv('AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/bc_features.csv') #save features as csv file
pd.DataFrame(breast_cancer_data.get('target')).to_csv('AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/bc_targets.csv') #save features as csv file
bc_features = pd.read_csv('AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/bc_features.csv')
bc_targets = pd.read_csv('AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/bc_targets.csv')

bc_features = bc_features.drop(bc_features.columns[0], axis=1) #drop first column
bc_targets = bc_targets.drop(bc_targets.columns[0], axis=1) #drop first column 
bc_train_X, bc_test_X, bc_train_y, bc_test_y = get_train_test_datasets(bc_features,bc_targets)

bc_train_X.to_csv('AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/train/bc_train_X.csv')
bc_train_y.to_csv('AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/train/bc_train_y_1.csv')
bc_test_X.to_csv('AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/test/bc_test_X.csv')
bc_test_y.to_csv('AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/test/bc_test_y.csv')

###convert dataframes to pickle####################################################################################
bc_train_y.to_pickle('AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/train/bc_train_y')
train_y = pd.read_pickle('AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/train/bc_train_y')


#Upload train and test directories of BC data to S3 using sagemaker api
train_input_bc = sagemaker_session.upload_data(path= 'AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/train',
                                            bucket='sklearn-sagemaker-data',
                                            key_prefix='breast_cancer/train')
test_input_bc = sagemaker_session.upload_data(path= 'AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/test',
                                            bucket='sklearn-sagemaker-data',
                                            key_prefix='breast_cancer/test')

train_input_breast_cancer = 's3://sklearn-sagemaker-data/breast_cancer/train'  #'/' is not given in the file name in the end since it will be joined to files using os.path.join
test_input_breast_cancer = 's3://sklearn-sagemaker-data/breast_cancer/test'

#Upload individual feature and target files to s3 bucket using s3client
s3client.upload_file(Filename='AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/bc_features.csv',  #complete filename in local dir
                     Bucket='sklearn-sagemaker-data', 
                     Key='breast_cancer/bc-features.csv') #complete path to file in s3 which is a file name

s3client.upload_file(Filename='AWS_Sagemaker/sklearn_sagemaker_deploy/data/breast_cancer/bc_targets.csv',  #complete filename in local dir
                     Bucket='sklearn-sagemaker-data', 
                     Key='breast_cancer/bc-targets.csv')


#Use this code to download specific file from s3 to local directory
#bc_s3_download = s3client.download_file(Bucket='sklearn-sagemaker-data',
#                                Key='breast_cancer/bc-features.csv', #complete path of file in s3
#                                Filename='AWS_Sagemaker/sklearn_sagemaker_deploy/bc_s3_features.csv') #complete path for file in local dir

# Below code not working for uploading .Bunch object to S3 directly
# s3client.put_object(Body=sklearn.datasets.load_breast_cancer(),
#                     Bucket='sklearn-sagemaker-data',
#                     key='sklearn-sagemaker-data/breast-cancer/bc-data.Bunch')

#Create sagemaker estimator for iris
from sagemaker.sklearn.estimator import SKLearn
FRAMEWORK_VERSION = "0.23-1"
script_path_iris = 'AWS_Sagemaker/sklearn_sagemaker_deploy/sklearn_script_iris.py'
script_path_breast_cancer = 'AWS_Sagemaker/sklearn_sagemaker_deploy/sklearn_script_breast_cancer.py'
sklearn_path_random_model = 'AWS_Sagemaker/sklearn_sagemaker_deploy/sklearn_script_random_model.py'

sklearn_estimator_iris = SKLearn(entry_point= script_path_iris,
                            framework_version= FRAMEWORK_VERSION,
                            instance_type= "ml.c4.xlarge",
                            role=SageMakerRole,
                            sagemaker_session= sagemaker_session,
                            hyperparameters={'max_leaf_nodes': 30})

sklearn_estimator_breast_cancer = SKLearn(entry_point= script_path_breast_cancer,
                            framework_version= FRAMEWORK_VERSION,
                            instance_type= "ml.c4.xlarge",
                            role=SageMakerRole,
                            sagemaker_session= sagemaker_session,
                            hyperparameters={'max_leaf_nodes': 30})

sklearn_estimator_random_model = SKLearn(entry_point= sklearn_path_random_model,
                            framework_version= FRAMEWORK_VERSION,
                            instance_type= "ml.c4.xlarge",
                            role=SageMakerRole,
                            sagemaker_session= sagemaker_session)

#This will start a SageMaker Training job that will download the 
# data for us, invoke our scikit-learn code (in the provided script 
# file), and save any model artifacts that the script creates.
sklearn_estimator_iris.fit({'train': train_input_iris})
sklearn_estimator_breast_cancer.fit({'train': train_input_breast_cancer}) #always provide directory of s3 training/testing data which are parsed for the training 

sklearn_estimator_random_model.fit({'train': train_input_breast_cancer, 'test': test_input_breast_cancer})

#Deploy the trained iris model to make inference requests
predictor_iris = sklearn_estimator_iris.deploy(initial_instance_count=1, instance_type="ml.m5.xlarge")
                                     

import itertools
import pandas as pd

shape = pd.read_csv("AWS_Sagemaker/sklearn_sagemaker_deploy/data/iris_data.csv", header=None)

a = [50*i for i in range(3)]
b = [40+i for i in range(10)]

indices = [i+j for i,j in itertools.product(a,b)]

test_data = shape.iloc[indices[:-1]]
test_X = test_data.iloc[:,1:]
test_y =test_data.iloc[:,0]

pred_y = predictor_iris.predict(test_X.values)
from sklearn.metrics import accuracy_score
print(accuracy_score(test_y,pred_y))

#Iris Endpoint cleanup
predictor_iris.delete_endpoint()

#Iris Batch transform jobs
##We can also use the trained model for asynchronous batch inference on S3 data using SageMaker Batch Transform.
transformer=sklearn_estimator_iris.transformer(instance_count=1,
                                          instance_type='ml.m5.large')
