from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from feature_engineering import feature_engineering
from dvc import dvc
def data_preprocess():
    dvc()
    data = feature_engineering()
    X = data.drop(['y_new'],axis=1)
    y = data['y_new']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
    print("Data os splitted to train and test......")
    return(X_train, X_test, y_train, y_test)

#data_preprocess() 