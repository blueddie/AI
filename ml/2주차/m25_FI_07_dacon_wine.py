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
print(x.shape)
# 하위 20퍼센트의 인덱스: [2 4 7] 
to_delete = [2, 4, 7]
for idx in sorted(to_delete, reverse=True):
    x = x.drop(x.columns[idx], axis=1)
print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=13, train_size=0.8, stratify=y)

#2
models = [DecisionTreeClassifier(random_state=777), RandomForestClassifier(random_state=777), GradientBoostingClassifier(random_state=777), XGBClassifier(random_state=777)]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(model.__class__.__name__, "acc :", results)  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")
    
    threshold = np.percentile(model.feature_importances_, 20)  
    low_importance_indices = np.where(model.feature_importances_ < threshold)[0]   
    print("하위 20퍼센트의 인덱스:", low_importance_indices, "\n")
    
# DecisionTreeClassifier acc : 0.5654545454545454
# DecisionTreeClassifier feature importance
# [0.06651335 0.10194018 0.07167512 0.07992314 0.09358499 0.10266196
#  0.10359001 0.06522528 0.08988639 0.08680043 0.1358018  0.00239736]
# 하위 20퍼센트의 인덱스: [ 0  7 11] 

# RandomForestClassifier acc : 0.6727272727272727
# RandomForestClassifier feature importance
# [0.07377122 0.09912928 0.08008338 0.08484769 0.08453191 0.08576396
#  0.09036342 0.1072045  0.08171513 0.08876355 0.12033334 0.00349261]
# 하위 20퍼센트의 인덱스: [ 0  2 11] 

# GradientBoostingClassifier acc : 0.5890909090909091
# GradientBoostingClassifier feature importance
# [0.03747928 0.15316603 0.04005025 0.06864724 0.0482672  0.06050426
#  0.06255885 0.06684848 0.04968547 0.08221996 0.32299837 0.00757462]
# 하위 20퍼센트의 인덱스: [ 0  2 11] 

# XGBClassifier acc : 0.6418181818181818
# XGBClassifier feature importance
# [0.05903795 0.08677507 0.05716294 0.06239884 0.05602742 0.06377011
#  0.05723355 0.05497356 0.05759696 0.06687249 0.16924419 0.20890692]
# 하위 20퍼센트의 인덱스: [2 4 7] 

#================================================
# DecisionTreeClassifier acc : 0.6090909090909091
# DecisionTreeClassifier feature importance
# [0.09348309 0.13780074 0.10542174 0.10592865 0.1421407  0.13223009
#  0.12493835 0.15436001 0.00369664]
# 하위 20퍼센트의 인덱스: [0 8]

# RandomForestClassifier acc : 0.6472727272727272
# RandomForestClassifier feature importance
# [0.10548039 0.13338852 0.11631166 0.1107535  0.12826859 0.11213483
#  0.11513844 0.1714325  0.00709156]
# 하위 20퍼센트의 인덱스: [0 8]

# GradientBoostingClassifier acc : 0.5763636363636364
# GradientBoostingClassifier feature importance
# [0.05903307 0.18572344 0.08225746 0.07077541 0.08696845 0.05857142
#  0.08646834 0.35979186 0.01041054]
# 하위 20퍼센트의 인덱스: [5 8]

# XGBClassifier acc : 0.63
# XGBClassifier feature importance
# [0.07705621 0.11743697 0.08334591 0.08594417 0.08570193 0.07727004
#  0.08663107 0.20070477 0.1859089 ]
# 하위 20퍼센트의 인덱스: [0 5]