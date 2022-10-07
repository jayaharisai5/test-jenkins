import boto3
import pandas as pd
import csv
s3 = boto3.client(
    's3',
    aws_access_key_id='AKIA3YG72WSKAY3DQARO',
    aws_secret_access_key='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',
    region_name='us-east-1'
) #1

obj = s3.get_object(Bucket='mlops-storage1', Key='hari/bank.csv') #2
data = obj['Body'].read().decode('utf-8').splitlines() #3
records = csv.reader(data) #4
headers = next(records) #5
print('headers: %s' % (headers)) 
for eachRecord in records: #6
    print(eachRecord)