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
from sklearn.ensemble import RandomForestRegressor


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

model_p = Perceptron()
model_l = LinearRegression()
model_K = KNeighborsRegressor()
model_D = DecisionTreeRegressor()
model_R = RandomForestRegressor()

models = [model_p, model_l , model_K, model_D, model_R]

for model in models :
    model.fit(X_train, y_train)
    results = model.score(X_test, y_test)
    print(model, " r2 :", results)  

# Perceptron()  r2 : 0.014634146341463415
# LinearRegression()  r2 : 0.5828110390170835
# KNeighborsRegressor()  r2 : 0.3930534526268208
# DecisionTreeRegressor()  r2 : 0.5028412833909639
# RandomForestRegressor()  r2 : 0.7981905560578936