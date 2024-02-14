from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1226, train_size=0.9)

#2
models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(f"{model.__class__.__name__} r2 : {results}")  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")

# DecisionTreeRegressor r2 : 0.26141883113448716
# DecisionTreeRegressor feature importance
# [0.06569145 0.01267365 0.24324975 0.09095084 0.04192964 0.08080296
#  0.05596445 0.02074769 0.31018778 0.07780179]
# RandomForestRegressor r2 : 0.6452230368485189
# RandomForestRegressor feature importance
# [0.06015149 0.01306268 0.27926982 0.10208574 0.04805207 0.062892
#  0.05597705 0.02537429 0.28397229 0.06916258]
# GradientBoostingRegressor r2 : 0.6338053498042353
# GradientBoostingRegressor feature importance
# [0.05116651 0.02518397 0.29454374 0.11337509 0.02557176 0.05352404
#  0.03898398 0.02794416 0.30366551 0.06604126]
# XGBRegressor r2 : 0.5183522914324592
# XGBRegressor feature importance
# [0.03997793 0.05603581 0.18781506 0.07385565 0.04766916 0.06244185
#  0.05147139 0.07944071 0.31233358 0.08895889]
