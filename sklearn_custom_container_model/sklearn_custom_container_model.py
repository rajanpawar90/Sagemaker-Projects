# -*- coding: utf-8 -*-
"""
Spyder Editor
https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb
This is a temporary script file.
"""
import sklearn
from sklearn.datasets import california_housing, load_breast_cancer, load_wine, load_iris
from sklearn.metrics import accuracy_score
import sagemaker
from sagemaker import TrainingInput
import boto3
import re
from sagemaker import image_uris
from sagemaker import TrainingInput
import os
import pandas as pd
import numpy as np

import sys
print(sys.path)
sys.path.append('C:\\Users\\Rajan\\Desktop\\Tech_Projects\\ML_projects')
#from ML_Utils.ML_Utils import *

region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
account = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
iam = boto3.client('iam')
s3client = boto3.client('s3')

SageMakerRole = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20200112T192472')['Role']['Arn']

iris_data = load_iris()
iris_features = pd.DataFrame(iris_data.get('data'))
iris_targets = pd.DataFrame(iris_data.get('target'))
iris_features.to_csv('sklearn_custom_container_model/data/iris_features.csv')

#pickle training dfs for sagemaker upload
pd.to_pickle(iris_features, 'sklearn_custom_container_model/data/train/iris_features.pkl')
pd.to_pickle(iris_targets, 'sklearn_custom_container_model/data/train/iris_targets.pkl')
    
#upload pickled training dfs to sagemaker
s3_data_location  = sagemaker_session.upload_data(path='sklearn_custom_container_model/data',
                              bucket='sklearn-sagemaker-data',
                              key_prefix='iris')

# Create an estimator and fit the model
image = '{}.dkr.ecr.{}.amazonaws.com/sagemaker-decision-trees'.format(account, region) #-decision:latest

tree = sagemaker.estimator.Estimator(image,
                                     SageMakerRole,
                                     instance_count=1,
                                     instance_type='ml.c4.2xlarge',
                                     output_path='s3://{}/output'.format(sagemaker_session.default_bucket()),
                                     sagemaker_session=sagemaker_session)

tree.fit(s3_data_location)

print(sagemaker_session.boto_session.region_name)
    
#Hosting your model
##Deploy the model
from sagemaker.predictor import csv_serializer
predictor = tree.deploy(1, 'ml.m4.xlarge', serializer=csv_serializer)

##Choose some data and use it for a prediction
shape = pd.read_csv('data/iris_features.csv', header=None)
shape.sample(3)
import itertools
a = [50*i for i in range(3)]
b = [40+i for i in range(10)]
indices = [i+j for i,j in itertools.product(a,b)]
test_data = shape.iloc[indices[:-1]]
print(predictor.predict(test_data.values).decode('utf-8'))

## Delete the endpoint
sagemaker_session.delete_endpoint(predictor.endpoint)

#Run a batch transform jon
##Create a transform job
transform_output_folder ='batch-transform-output'
output_path = 's3://{}/{}'.format(sagemaker_session.default_bucket(), transform_output_folder)

transformer = tree.transformer(instance_count=1,
                               instance_type='ml.m4.xlarge',
                               output_path=output_path,
                               assemble_with='Line',
                               accept='text/csv')

transformer.transform(s3_data_location,
                      content_type='text/csv',
                      split_type='Line',
                      input_filter='$[1:]')
transformer.wait()

##View output
s3_client = sagemaker_session.boto_session.client('s3')
s3_client.download_file(sagemaker_session.default_bucket(),
                        '{}/iris.csv.output'.format(transform_output_folder), '/tmp/iris.csv.out')
with open('tmp/iris.csv.out') as f:
    results = f.readlines()
print("Transform results: \n{}".format(''.join(results)))