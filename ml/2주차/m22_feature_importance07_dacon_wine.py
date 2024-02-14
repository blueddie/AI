# https://dacon.io/competitions/open/235610/overview/description
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder

#1.
path = "C://_data//dacon//wine//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

items = train_csv['type']

encoder = LabelEncoder()
encoder.fit(items)
train_csv['type'] = encoder.transform(items)
test_csv['type'] = encoder.transform(test_csv['type'])
# print('인코당 클래스: ', encoder.classes_)  #['red' 'white']


x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
# y = y.values.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# ohe.fit(y)
# y = ohe.transform(y)
# print(y)


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
    
# DecisionTreeClassifier acc : 0.5709090909090909
# DecisionTreeClassifier feature importance
# [0.06492529 0.10573071 0.07172961 0.07739232 0.08892866 0.09937108
#  0.10196243 0.07702188 0.09545148 0.08288317 0.13265793 0.00194543]
# RandomForestClassifier acc : 0.6618181818181819
# RandomForestClassifier feature importance
# [0.07451206 0.10096315 0.07949961 0.08410303 0.08478071 0.08542744
#  0.09138799 0.10104385 0.08200736 0.08836673 0.12445439 0.00345368]
# GradientBoostingClassifier acc : 0.59
# GradientBoostingClassifier feature importance
# [0.03737244 0.15273088 0.04030272 0.06837106 0.04834534 0.06069732
#  0.06115094 0.06703002 0.04969091 0.08248759 0.3242653  0.00755546]
# XGBClassifier acc : 0.6418181818181818
# XGBClassifier feature importance
# [0.05903795 0.08677507 0.05716294 0.06239884 0.05602742 0.06377011
#  0.05723355 0.05497356 0.05759696 0.06687249 0.16924419 0.20890692]