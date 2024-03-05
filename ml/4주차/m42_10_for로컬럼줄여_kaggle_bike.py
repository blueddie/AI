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

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=15)
#2. 모델
models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for model in models :
    
    #3 
    model.fit(X_train, y_train)

    #4
    results = model.score(X_test, y_test)
    print(f"{model.__class__.__name__} r2 : {results}")  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}") 

# DecisionTreeRegressor r2 : -0.1365564618155357
# DecisionTreeRegressor feature importance
# [0.06475823 0.00719154 0.03471496 0.05277062 0.14707136 0.24886639
#  0.2465135  0.19811341]
# RandomForestRegressor r2 : 0.3061030883018778
# RandomForestRegressor feature importance
# [0.0672884  0.00658393 0.03982755 0.05170935 0.14575174 0.24079888
#  0.25341718 0.19462298]
# GradientBoostingRegressor r2 : 0.33665962275592387
# GradientBoostingRegressor feature importance
# [0.07582306 0.00187774 0.03236003 0.0139478  0.18705326 0.32466745
#  0.34315761 0.02111304]
# XGBRegressor r2 : 0.3319260315586371
# XGBRegressor feature importance
# [0.12161168 0.0490254  0.0985127  0.07099905 0.11209968 0.34708703
#  0.14419632 0.05646813]