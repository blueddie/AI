# https://dacon.io/competitions/open/235576/codeshare/6969?page=1&dtype=recent // 대회 주소
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


#1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv")   #, index_col=0
print(submission_csv)

X = train_csv.drop(['count'], axis=1)

y = train_csv["count"]

test_csv = test_csv.fillna(test_csv.mean())

X['hour_bef_temperature'] = X['hour_bef_temperature'].fillna(X['hour_bef_temperature'].mean())

X.loc[[1420],['hour_bef_precipitation']] = 0.0
X.loc[[1553],['hour_bef_precipitation']] = 1.0

X.loc[[1420],['hour_bef_humidity']] = 37.0
X.loc[[1553],['hour_bef_humidity']] = 82.0

X['hour_bef_visibility'] = X['hour_bef_visibility'].fillna(X['hour_bef_visibility'].mean())

X = X.fillna(X.mean())
# first_name = "submission_0110_es_rs_"
second_name = ".csv"

print(X.shape)  #(1459, 9)


X = X.astype(np.float32)
test_csv = test_csv.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.86, random_state=56238592)

models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for model in models :
    
    #3 
    model.fit(X_train, y_train)

    #4
    results = model.score(X_test, y_test)
    print(f"{model.__class__.__name__} r2 : {results}")  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}") 

# DecisionTreeRegressor r2 : 0.46657653886294925
# DecisionTreeRegressor feature importance
# [0.5994146  0.20241907 0.03062735 0.03031961 0.03304757 0.02480632
#  0.03091098 0.03208256 0.01637194]
# RandomForestRegressor r2 : 0.7974781054411997
# RandomForestRegressor feature importance
# [0.58451914 0.18041203 0.02007406 0.03162561 0.04069943 0.03727088
#  0.04589572 0.03477558 0.02472754]
# GradientBoostingRegressor r2 : 0.7978703262995086
# GradientBoostingRegressor feature importance
# [0.64167762 0.21007434 0.02177372 0.01501688 0.01859209 0.03024778
#  0.03010691 0.02018728 0.01232338]
# XGBRegressor r2 : 0.7662079535353098
# XGBRegressor feature importance
# [0.31667817 0.09801537 0.41667515 0.02449488 0.03325202 0.02678071
#  0.03088947 0.03081557 0.02239868]