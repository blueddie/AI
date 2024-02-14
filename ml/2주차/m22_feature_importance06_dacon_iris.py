# https://dacon.io/competitions/open/236070/overview/description
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1.
path = "C://_data//dacon//iris//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

x = train_csv.drop(['species'], axis=1)
y = train_csv['species']

# print(X.shape)  #(120, 4)
# print(y.shape)  #(120,)
print(pd.value_counts(y))



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=13, train_size=0.8, stratify=y)

#2
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(model.__class__.__name__, "acc :", results)  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")

# DecisionTreeClassifier acc : 0.9583333333333334
# DecisionTreeClassifier feature importance
# [0.         0.         0.58493719 0.41506281]
# RandomForestClassifier acc : 0.9583333333333334
# RandomForestClassifier feature importance
# [0.08594053 0.03991902 0.45117583 0.42296462]
# GradientBoostingClassifier acc : 0.9583333333333334
# GradientBoostingClassifier feature importance
# [0.00107802 0.00261454 0.4619309  0.53437654]
# XGBClassifier acc : 0.9583333333333334
# XGBClassifier feature importance
# [0.01898877 0.01431537 0.6825475  0.2841484 ]