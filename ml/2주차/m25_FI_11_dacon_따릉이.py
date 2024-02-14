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

x = X.fillna(X.mean())
# first_name = "submission_0110_es_rs_"
second_name = ".csv"

print(X.shape)  #(1459, 9)


x = x.astype(np.float32)
test_csv = test_csv.astype(np.float32)

# 하위 20퍼센트의 인덱스: [3 8]
to_delete = [3, 8]
for idx in sorted(to_delete, reverse=True):
    X = X.drop(X.columns[idx], axis=1)
print(X.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86, random_state=56238592)

models = [DecisionTreeRegressor(random_state=777), RandomForestRegressor(random_state=777), GradientBoostingRegressor(random_state=777), XGBRegressor(random_state=777)]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(model.__class__.__name__, "r2 :", results)  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")
    threshold = np.percentile(model.feature_importances_, 20)  
    low_importance_indices = np.where(model.feature_importances_ < threshold)[0]   
    print("하위 20퍼센트의 인덱스:", low_importance_indices, "\n")

# DecisionTreeRegressor r2 : 0.4724281953120024
# DecisionTreeRegressor feature importance
# [0.59513894 0.20030078 0.03063467 0.034189   0.03494647 0.02704101
#  0.03145001 0.03019809 0.01610103]
# 하위 20퍼센트의 인덱스: [5 8]

# RandomForestRegressor r2 : 0.7972627006214312
# RandomForestRegressor feature importance
# [0.58150363 0.18616057 0.02129172 0.03096562 0.04065937 0.03546288
#  0.04777776 0.03267108 0.02350736]
# 하위 20퍼센트의 인덱스: [2 8]

# GradientBoostingRegressor r2 : 0.797322970045593
# GradientBoostingRegressor feature importance
# [0.64184318 0.21036852 0.02177372 0.01519011 0.01856067 0.02993719
#  0.03027866 0.01979203 0.01225592]
# 하위 20퍼센트의 인덱스: [3 8]

# XGBRegressor r2 : 0.7662079535353098
# XGBRegressor feature importance
# [0.31667817 0.09801537 0.41667515 0.02449488 0.03325202 0.02678071
#  0.03088947 0.03081557 0.02239868]
# 하위 20퍼센트의 인덱스: [3 8]
#=================================================================================
# DecisionTreeRegressor r2 : 0.4724281953120024
# DecisionTreeRegressor feature importance
# [0.59513894 0.20030078 0.03063467 0.034189   0.03494647 0.02704101
#  0.03145001 0.03019809 0.01610103]
# 하위 20퍼센트의 인덱스: [5 8] 

# RandomForestRegressor r2 : 0.7972627006214312
# RandomForestRegressor feature importance
# [0.58150363 0.18616057 0.02129172 0.03096562 0.04065937 0.03546288
#  0.04777776 0.03267108 0.02350736]
# 하위 20퍼센트의 인덱스: [2 8]

# GradientBoostingRegressor r2 : 0.797322970045593
# GradientBoostingRegressor feature importance
# [0.64184318 0.21036852 0.02177372 0.01519011 0.01856067 0.02993719
#  0.03027866 0.01979203 0.01225592]
# 하위 20퍼센트의 인덱스: [3 8]

# XGBRegressor r2 : 0.7662079535353098
# XGBRegressor feature importance
# [0.31667817 0.09801537 0.41667515 0.02449488 0.03325202 0.02678071
#  0.03088947 0.03081557 0.02239868]
# 하위 20퍼센트의 인덱스: [3 8]