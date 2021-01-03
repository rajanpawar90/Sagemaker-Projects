# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
from ML_Utils.ML_Utils import *
import sklearn
from sklearn.datasets import california_housing, load_breast_cancer, load_wine

wine_data = sklearn.datasets.load_wine()
wine_features = wine_data.get('data')
wine_targets = wine_data.get('target')
wine_X_train, wine_X_test, wine_y_train, wine_y_test = get_train_test_datasets(wine_features, wine_targets)                                                                           

import sagemaker 
import boto3
import re

#sample notebooks reviewed
# https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_randomforest/Sklearn_on_SageMaker_end2end.ipynb 
# https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_iris/scikit_learn_estimator_example_with_batch_transform.ipynb
# https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_inference_pipeline/Inference%20Pipeline%20with%20Scikit-learn%20and%20Linear%20Learner.ipynb

from sagemaker import image_uris
from sagemaker import TrainingInput
region = boto3.Session().region_name
container = image_uris.retrieve('image-classification', region)
bucket= 'your-bucket-name'
prefix = 'output'
#SageMakerRole='arn:aws:iam::xxxxxxxxxx:role/service-role/AmazonSageMaker-ExecutionRole-20191208T093742'

sagemaker_session = sagemaker.Session()
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    SageMakerRole = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20200112T192472')['Role']['Arn']

s3_output_location= 's3://{}/output'.format(bucket, prefix)

#Create an estimator for classification of images
classifier = sagemaker.estimator.Estimator(container,
                                           role=SageMakerRole,
                                           instance_count=1,
                                           instance_type='ml.t2.medium',
                                           volume_size = 50,
                                           max_run = 360000,
                                           input_mode= 'File',
                                           output_path=s3_output_location)

classifier.set_hyperparameters(num_layers=152,
                               use_pretrained_model=0,
                               image_shape='3,224,224',
                               num_classes=2,
                               mini_batch_size=32,
                               epochs=30,
                               learning_rate = 0.01,
                               num_training_samples = 963,
                               precision_dtype=  'float32')
    
#Training channels

from sagemaker import s3train

train_data = TrainingInput(s3train, distribution='FullyReplicated', 
                        content_type='application/x-image', s3_data_type='S3Prefix')
validation_data = TrainingInput(s3validation, distribution='FullyReplicated', 
                             content_type='application/x-image', s3_data_type='S3Prefix')
train_data_lst = TrainingInput(s3train_lst, distribution='FullyReplicated', 
                        content_type='application/x-image', s3_data_type='S3Prefix')
validation_data_lst = TrainingInput(s3validation_lst, distribution='FullyReplicated', 
                             content_type='application/x-image', s3_data_type='S3Prefix')

data_channels = {'train':train_data, 'validation':validation_data,
                 'train_lst':train_data_lst, 'validation_lst':validation_data_lst}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
