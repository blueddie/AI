# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import random
import time
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

X = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
test_csv = test_csv.drop([], axis=1)

# 하위 20퍼센트의 인덱스: [1 3]
to_delete = [1, 3]
for idx in sorted(to_delete, reverse=True):
    X = X.drop(X.columns[idx], axis=1)
print(X.shape)



x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=15)
#2. 모델
models = [DecisionTreeRegressor(random_state=777), RandomForestRegressor(random_state=777), GradientBoostingRegressor(random_state=777), XGBRegressor(random_state=777)]

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

# DecisionTreeRegressor acc : -0.12999716914505122
# DecisionTreeRegressor feature importance
# [0.06407618 0.0071613  0.03692302 0.05333034 0.15526512 0.24219261
#  0.24221862 0.19883281]
# 하위 20퍼센트의 인덱스: [1 2]

# RandomForestRegressor acc : 0.29992630162284906
# RandomForestRegressor feature importance
# [0.06798782 0.00660176 0.03992291 0.05069555 0.14502789 0.23968561
#  0.25385434 0.19622413]
# 하위 20퍼센트의 인덱스: [1 2] 

# GradientBoostingRegressor acc : 0.3366697248721946
# GradientBoostingRegressor feature importance
# [0.07582097 0.00187774 0.03236122 0.01391141 0.18659605 0.32515925
#  0.34320052 0.02107284]
# 하위 20퍼센트의 인덱스: [1 3]

# XGBRegressor acc : 0.3319260315586371
# XGBRegressor feature importance
# [0.12161168 0.0490254  0.0985127  0.07099905 0.11209968 0.34708703
#  0.14419632 0.05646813]
# 하위 20퍼센트의 인덱스: [1 7]
#==================================================================
# DecisionTreeRegressor acc : -0.08447421901143137
# DecisionTreeRegressor feature importance
# [0.0735528  0.04032046 0.14853401 0.26761854 0.26046105 0.20951314]
# 하위 20퍼센트의 인덱스: [1]

# RandomForestRegressor acc : 0.2666490428374715
# RandomForestRegressor feature importance
# [0.07160407 0.04311658 0.1528953  0.25010581 0.26821296 0.21406529]
# 하위 20퍼센트의 인덱스: [1] 

# GradientBoostingRegressor acc : 0.33444890578319864
# GradientBoostingRegressor feature importance
# [0.08008491 0.03401838 0.18996456 0.32717473 0.34497967 0.02377774]
# 하위 20퍼센트의 인덱스: [5]

# XGBRegressor acc : 0.3120836691403296
# XGBRegressor feature importance
# [0.13521498 0.10241656 0.13137558 0.40354106 0.16195595 0.0654958 ]
# 하위 20퍼센트의 인덱스: [5]