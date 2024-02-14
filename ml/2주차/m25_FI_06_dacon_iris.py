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
print(x.shape)
# print(X.shape)  #(120, 4)
# print(y.shape)  #(120,)
print(pd.value_counts(y))
# 하위 20퍼센트의 인덱스: [1]
to_delete = [1]
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
    
# DecisionTreeClassifier acc : 0.9583333333333334
# DecisionTreeClassifier feature importance
# [0.         0.         0.58493719 0.41506281]
# 하위 20퍼센트의 인덱스: []

# RandomForestClassifier acc : 0.9583333333333334
# RandomForestClassifier feature importance
# [0.09560565 0.03676408 0.42971759 0.43791267]
# 하위 20퍼센트의 인덱스: [1]

# GradientBoostingClassifier acc : 0.9583333333333334
# GradientBoostingClassifier feature importance
# [0.00155996 0.00261589 0.62852601 0.36729815]
# 하위 20퍼센트의 인덱스: [0]

# XGBClassifier acc : 0.9583333333333334
# XGBClassifier feature importance
# [0.01898877 0.01431537 0.6825475  0.2841484 ]
# 하위 20퍼센트의 인덱스: [1]

#===============================================================
# DecisionTreeClassifier acc : 0.9583333333333334
# DecisionTreeClassifier feature importance
# [0.02084012 0.06377144 0.91538844]
# 하위 20퍼센트의 인덱스: [0] 

# RandomForestClassifier acc : 0.9583333333333334
# RandomForestClassifier feature importance
# [0.18294517 0.46906974 0.34798508]
# 하위 20퍼센트의 인덱스: [0] 

# GradientBoostingClassifier acc : 0.9583333333333334
# GradientBoostingClassifier feature importance
# [0.00149886 0.4750269  0.52347423]
# 하위 20퍼센트의 인덱스: [0] 

# XGBClassifier acc : 0.9583333333333334
# XGBClassifier feature importance
# [0.01983006 0.7554812  0.22468878]
# 하위 20퍼센트의 인덱스: [0] 