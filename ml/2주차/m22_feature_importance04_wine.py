from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
#1.
datasets = load_wine()

X = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=13, train_size=0.8, stratify=y)

models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(model.__class__.__name__, "acc :", results)  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")
    
# DecisionTreeClassifier acc : 0.9166666666666666
# DecisionTreeClassifier feature importance
# [0.         0.         0.02086548 0.         0.         0.
#  0.40558571 0.         0.         0.38697541 0.         0.04111597
#  0.14545744]
# RandomForestClassifier acc : 1.0
# RandomForestClassifier feature importance
# [0.11437239 0.02226063 0.01260704 0.0315248  0.03189473 0.03661207
#  0.14561573 0.01243515 0.03013139 0.15818863 0.08748967 0.13011847
#  0.18674929]
# GradientBoostingClassifier acc : 0.9722222222222222
# GradientBoostingClassifier feature importance
# [3.13758059e-03 5.46874863e-02 7.31177718e-03 2.02774932e-03
#  8.03524212e-03 2.16595985e-03 1.92923410e-01 2.26128218e-04
#  7.34307442e-04 3.03308981e-01 2.43953367e-02 1.35307662e-01
#  2.65738379e-01]
# XGBClassifier acc : 0.9444444444444444
# XGBClassifier feature importance
# [0.01018853 0.05218591 0.0091292  0.00567487 0.01885441 0.0277835
#  0.14115812 0.01073536 0.00816022 0.18099341 0.04160717 0.3505479
#  0.14298138]