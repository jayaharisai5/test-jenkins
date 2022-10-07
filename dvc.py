import boto3
import pandas as pd
import numpy as np
import csv
import os
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

def dvc():
    upload_files("cleaned_data.csv", "mlops-storage1")
    os.system("git init")
    os.system("dvc init -f")
    os.system("dvc remote add -d dvc-remote s3://mlops-storage1/dvc")
    os.system("dvc add cleaned_data.csv")
    os.system("git add cleaned_data.csv.dvc .gitignore")
    print("dvc file is pushing to s3......")
    os.system("dvc push")
    print("done")


    print("Deleting the unwanted files......")
    os.remove("cleaned_data.csv")
    os.remove("cleaned_data.csv.dvc")
    print("DVC Done")