from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import datetime
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8)

#2
models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(f"{model.__class__.__name__} r2 : {results}")  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}") 

# DecisionTreeRegressor r2 : 0.6001035557167496
# DecisionTreeRegressor feature importance
# [0.51451892 0.05270302 0.05161624 0.02842308 0.03156616 0.1354897
#  0.09358738 0.09209549]
# RandomForestRegressor r2 : 0.8186977998993965
# RandomForestRegressor feature importance
# [0.51904506 0.05319631 0.04723161 0.02985428 0.031571   0.13602753
#  0.09151922 0.09155499]
# GradientBoostingRegressor r2 : 0.7932709240404006
# GradientBoostingRegressor feature importance
# [0.59787189 0.03007531 0.02433379 0.004659   0.00268903 0.12193307
#  0.09591751 0.12252039]
# XGBRegressor r2 : 0.8386188269983407
# XGBRegressor feature importance
# [0.48721272 0.06797588 0.05002455 0.02223908 0.02304351 0.13496281
#  0.10358929 0.11095219]