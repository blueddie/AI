import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


#1. 데이터
datasets = load_breast_cancer()

X = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=53, train_size=0.8, stratify=y)

#2 모델
# model = LinearSVC(C=100)        # C가 크면 training포인트를 정확히 구분(굴곡지다), C가 작으면 직선에 가깝다.
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(model.__class__.__name__, "acc :", results)  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")
    
# DecisionTreeClassifier acc : 0.9122807017543859
# DecisionTreeClassifier feature importance
# [0.         0.04299701 0.00626075 0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.04125562 0.0022404  0.         0.01760836 0.
#  0.         0.01408669 0.         0.03742135 0.69925315 0.01576672
#  0.00932356 0.01811146 0.         0.09567494 0.         0.        ]
# RandomForestClassifier acc : 0.9649122807017544
# RandomForestClassifier feature importance
# [0.02983656 0.01183896 0.04076089 0.05106236 0.00569148 0.00523903
#  0.04455282 0.10714978 0.00441927 0.00274717 0.00920409 0.00501033
#  0.01004834 0.02765508 0.00377432 0.00334234 0.00432161 0.00416023
#  0.00358639 0.00599841 0.12419539 0.01565628 0.22828686 0.10014622
#  0.01829904 0.02150172 0.01886845 0.07398958 0.0082709  0.0103861 ]
# GradientBoostingClassifier acc : 0.9736842105263158
# GradientBoostingClassifier feature importance
# [1.31939689e-03 2.16614680e-02 3.49963155e-04 8.98496501e-03
#  4.91159101e-03 1.52045028e-03 5.44231906e-03 3.11837292e-02
#  5.47396883e-04 6.78143989e-07 4.37814891e-03 9.26779047e-03
#  8.34925960e-04 4.67776849e-03 1.60619959e-03 1.34095412e-03
#  1.36740100e-03 3.09107940e-03 0.00000000e+00 6.05753186e-04
#  2.42436507e-01 4.84420633e-02 4.56792722e-01 2.98629691e-02
#  1.99855514e-02 3.57992756e-04 3.46457647e-03 9.25276067e-02
#  1.43029692e-03 1.60773605e-03]
# XGBClassifier acc : 0.9824561403508771
# XGBClassifier feature importance
# [2.1164339e-02 1.8960079e-02 0.0000000e+00 3.4322668e-02 4.9383924e-03
#  2.8294218e-03 7.7569075e-03 8.0713026e-02 1.2768973e-03 4.1659730e-03
#  6.6776555e-03 2.4561500e-03 2.1841312e-03 5.6698131e-03 2.1871186e-03
#  4.1694804e-03 7.2960863e-03 2.5183205e-02 1.3305614e-03 2.8682374e-03
#  2.7113535e-02 1.8029649e-02 5.4660314e-01 2.5041355e-02 8.8974154e-03
#  4.0043134e-04 1.9765066e-02 1.0810049e-01 3.0872624e-03 6.8114665e-03]