#impring the required libraries
import pandas as pd  #Data manipulation
import numpy as np   #Handling with array and scientific calculation
import warnings
import pickle
#improts from the sklearn
import sklearn.ensemble as es
from xgboost import XGBRegressor
import sklearn.tree as tr
import sklearn.linear_model as lm
import sklearn.neighbors as nbs
import sklearn.dummy as d
import sklearn.metrics as mt
import math
import warnings
warnings.filterwarnings("ignore")

### For Classification model

#importing the models from sklearn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
# Mentioning the models
dtc = tree.DecisionTreeClassifier()
rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()
lr = LogisticRegression()
rc = RidgeClassifier()
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
bnb = BernoulliNB()
adc = AdaBoostClassifier()
etc = ExtraTreesClassifier()
knc = KNeighborsClassifier()
dummy_clf = DummyClassifier()
svc = LinearSVC()

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score




# Mentioning the models
etr = es.ExtraTreesRegressor()
xgb = XGBRegressor()
rfr = es.RandomForestRegressor()
lr = lm.LinearRegression()
gbr = es.GradientBoostingRegressor()
dtr = tr.DecisionTreeRegressor()
knr = nbs.KNeighborsRegressor()
abr = es.AdaBoostRegressor()
lsr = lm.Lasso()
llr = lm.LassoLars()
rr = lm.Ridge()
br = lm.BayesianRidge()
hr = lm.HuberRegressor()
omp = lm.OrthogonalMatchingPursuit()
lar = lm.Lars()
en = lm.ElasticNet()
par = lm.PassiveAggressiveRegressor()
dr = d.DummyRegressor()


### Function to trian the best_model in the Classification Model
def cla_model_training(x_train, x_test, y_train, y_test):
    collect = []
    str_algo = []
    algo = [dtc, rfc, gbc, lr, rc, lda, qda, bnb, adc, etc, knc, dummy_clf, svc]
    for i in algo:
        a = []
        train = i.fit(x_train,y_train)
        score = train.score(x_test,y_test)
        a.append(score)
        s = str(i)
        str_algo.append(s)
        collect.append(a)
    dataframe = pd.DataFrame(np.column_stack([str_algo, collect]), 
                               columns=['Algorithms', 'Scores'])
    
    max_value= max(collect)
    for i in range(len(collect)):
        if collect[i] == max_value:
            best_model = algo[i]
    
    y_predict = best_model.predict(x_test)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    f1score = f1_score(y_test, y_predict)

    with open("best_model_classification.pkl", 'wb') as p:          #Creating a pickle file for the bets model
        pickle.dump(best_model, p)
    
    return  dataframe, best_model, precision, recall, accuracy,f1score 

### Function to train the best_model in the Regression Model
def reg_model_train(x_train, x_test, y_train, y_test):
    algorithms = [etr, xgb, rfr, lr, gbr, dtr, knr, abr, lsr, llr, rr, br, hr, omp, lar, en, par, dr]
    algo_names = ['ExtraTreesRegressor', 'XGBRegressor', 'RandomForestRegressor', 'LinearRegression', 'GradientBoostingRegressor', 'DecisionTreeRegressor', 'KNeighborsRegressor','AdaBoostRegressor', 'Lasso', 'LassoLars', 'Ridge', 'BayesianRidge', 'HuberRegressor', 'OrthogonalMatchingPursuit', 'Lars', 'ElasticNet', 'PassiveAggressiveRegressor', 'DummyRegressor'  ]
    al_scores = []
    for al in algorithms :
        a = []
        k = al.fit(x_train,y_train)
        score = k.score (x_test,y_test)
        a.append(score)
        al_scores.append(a)
    dataframe = pd.DataFrame(np.column_stack([algo_names, al_scores]), 
                               columns=['Algorithms', 'Scores'])

    max_score = max(al_scores)
    for s in range(len(al_scores)):
        if al_scores[s] == max_score:
            best_model = algorithms[s]
            
    y_predict = best_model.predict(x_test)

    with open("best_model_regression.pkl", 'wb') as p:              #Creating a pickle file for the bets model
        pickle.dump(best_model, p)

    MSE = mt.mean_squared_error(y_test, y_predict)
    RMSE = math.sqrt(MSE)
    MAE = mt.mean_absolute_error(y_test,y_predict)
        
    return dataframe, best_model, MSE, RMSE, MAE 