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
import numpy as np
import pandas as pd
#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target


# DataFrame
columns = datasets.feature_names
x = pd.DataFrame(x, columns=columns)
y = pd.Series(y)

# print(x)
# 하위 20퍼센트의 인덱스: [1 4]
to_delete = [1, 4]
for idx in sorted(to_delete, reverse=True):
    x = x.drop(x.columns[idx], axis=1)
print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1226, train_size=0.9)

#2
models = [DecisionTreeRegressor(random_state=777), RandomForestRegressor(random_state=777), GradientBoostingRegressor(random_state=777), XGBRegressor(random_state=777)]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(f"{model.__class__.__name__} r2 : {results}")  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")
    
    threshold = np.percentile(model.feature_importances_, 20)  
    low_importance_indices = np.where(model.feature_importances_ < threshold)[0]   
    print("하위 20퍼센트의 인덱스:", low_importance_indices)
    print()



# DecisionTreeRegressor r2 : 0.31233343960349125
# DecisionTreeRegressor feature importance
# [0.06392521 0.0146342  0.23144517 0.09904531 0.03781143 0.08338189
#  0.05281373 0.02538613 0.31557461 0.07598232]
# 하위 20퍼센트의 인덱스: [1 7]

# RandomForestRegressor r2 : 0.6307234568521958
# RandomForestRegressor feature importance
# [0.06259561 0.01326493 0.26635211 0.09979742 0.05109769 0.06084988
#  0.05325806 0.02530051 0.29473435 0.07274944]
# 하위 20퍼센트의 인덱스: [1 7]

# GradientBoostingRegressor r2 : 0.6346547430190066
# GradientBoostingRegressor feature importance
# [0.05163604 0.02473903 0.29443405 0.11323611 0.02661277 0.05354773
#  0.03792566 0.02864223 0.30335144 0.06587495]
# 하위 20퍼센트의 인덱스: [1 4]

# XGBRegressor r2 : 0.5183522914324592
# XGBRegressor feature importance
# [0.03997793 0.05603581 0.18781506 0.07385565 0.04766916 0.06244185
#  0.05147139 0.07944071 0.31233358 0.08895889]
# 하위 20퍼센트의 인덱스: [0 4]
#==========================================================================
# DecisionTreeRegressor r2 : 0.2016318249174821
# DecisionTreeRegressor feature importance
# [0.0740214  0.2478039  0.09833927 0.09845262 0.05819154 0.0190851
#  0.32697571 0.07713046]
# 하위 20퍼센트의 인덱스: [4 5]

# RandomForestRegressor r2 : 0.6473716875544879
# RandomForestRegressor feature importance
# [0.07105267 0.27128307 0.10608367 0.08523782 0.05708521 0.02734342
#  0.30232605 0.07958809]
# 하위 20퍼센트의 인덱스: [4 5]

# GradientBoostingRegressor r2 : 0.6366294162097621
# GradientBoostingRegressor feature importance
# [0.05285713 0.30456132 0.11722486 0.06988726 0.04920629 0.02313183
#  0.31196299 0.07116831]
# 하위 20퍼센트의 인덱스: [4 5]

# XGBRegressor r2 : 0.5776259925046024
# XGBRegressor feature importance
# [0.0497333  0.19169496 0.09762253 0.06692384 0.08209202 0.06828427
#  0.32127056 0.12237848]
# 하위 20퍼센트의 인덱스: [0 3]