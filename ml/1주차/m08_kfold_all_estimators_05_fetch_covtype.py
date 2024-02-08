from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, KFold

import warnings
warnings.filterwarnings('ignore')
#1

datasets = fetch_covtype()

X = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=15152, train_size=0.8)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgoritms : ', allAlgorithms)
print('모델의 갯수',len(allAlgorithms)) #모델의 갯수 41

for name, algorithm in allAlgorithms:
    
    try:
        #2 모델
        model = algorithm()
        #3 훈련
                #3 훈련
        scores = cross_val_score(model, x_train, y_train, cv=kfold)  #cv 교차검증
        print(f'=================={name}========================')        
        print(f'ACC : {scores}\n평균 ACC: {round(np.mean(scores), 4)}')

        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        # print(y_predict)
        # print(y_test)
        acc = accuracy_score(y_test, y_predict)
        print(f'cross_val_predict ACC : {acc} ')
    except Exception as e:
        # print(name , '에러 발생', e)
        continue