from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_preprocessing import data_preprocess
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_preprocessing import data_preprocess
from Auto_model_selection import cla_model_training
import pandas as pd
import pickle
import os

import boto3
s3 = boto3.client(
    's3',
    aws_access_key_id='AKIA3YG72WSKAY3DQARO',
    aws_secret_access_key='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',
    region_name='us-east-1'
) #1


def upload_files(file_name, bucket, object_name=None, args=None):
    if object_name is None:
        object_name=file_name
    response=s3.upload_file(file_name, bucket, object_name, ExtraArgs=args)
    print(response)


def model_selection():
    X_train, X_test, y_train, y_test = data_preprocess()
    cla_model_training(X_train, X_test, y_train, y_test)
    '''
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    decision_tree= dt.score(X_test, y_test)
    rt = RandomForestClassifier(n_estimators=100, n_jobs=1)
    rt.fit(X_train, y_train)
    random_forest = rt.score(X_test, y_test)
    lr= make_pipeline(StandardScaler(),LogisticRegression())
    lr.fit(X_train, y_train)
    logistic_regression = lr.score(X_test, y_test)
    results = [decision_tree, random_forest, logistic_regression ]
    maximum = max(results)
    if maximum == decision_tree:
        a = decision_tree
        b = dt
    elif maximum == random_forest:
        a = random_forest
        b = rt
    else:
        a = logistic_regression
        b = lr
    print(a,b)
    b.fit(X_train, y_train)
    
    filename = 'finalised_model.pkl'
    pickle.dump(b,open(filename,'wb'))
    upload_files("finalised_model.pkl", "mlops-storage1")
    '''
    '''rt5y6uio
    loaded_model = pickle.load(open(filename,'rb'))
    result1 = loaded_model.score(X_test, y_test)
    result2 = loaded_model.predict(X_test)
    f1_score = f1_score(y_test, result2)
    print(result1, result2)
    print(f1_score)

    upload_files("finalised_model.pkl", "mlops-storage1")
    os.remove("finalised_model.pkl")
    '''

model_selection()