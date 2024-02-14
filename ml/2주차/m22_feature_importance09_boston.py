from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
import datetime
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


datasets = load_boston()

x = datasets.data
y = datasets.target
# print(x.shape)  #(506, 13)
# print(y)  #(506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8)

models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(f"{model.__class__.__name__} r2 : {results}")  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")

# DecisionTreeRegressor r2 : 0.8345585036565907
# DecisionTreeRegressor feature importance
# [2.31509682e-02 2.80063506e-04 7.47077418e-03 9.65077906e-04
#  3.12146239e-02 2.44190574e-01 1.26919121e-02 8.93009047e-02
#  3.58874965e-03 1.12212236e-02 2.79701829e-02 1.06466399e-02
#  5.37308305e-01]
# RandomForestRegressor r2 : 0.9122293509878622
# RandomForestRegressor feature importance
# [0.034686   0.0007194  0.00596229 0.00087482 0.02906699 0.36495751
#  0.0168494  0.07740391 0.00350755 0.01478383 0.01470317 0.01103733
#  0.4254478 ]
# GradientBoostingRegressor r2 : 0.9296773006918521
# GradientBoostingRegressor feature importance
# [1.55576589e-02 2.57660486e-04 3.13686473e-03 1.20748480e-03
#  4.05323450e-02 2.98251011e-01 7.14077770e-03 1.01974758e-01
#  2.41779251e-03 1.49208122e-02 3.53566535e-02 8.79161089e-03
#  4.70454570e-01]
# XGBRegressor r2 : 0.9022178792512646
# XGBRegressor feature importance
# [0.01257844 0.00061468 0.01847021 0.00281567 0.04047003 0.1499025
#  0.00947146 0.10018896 0.01814521 0.03351607 0.02434715 0.00769892
#  0.58178073]
