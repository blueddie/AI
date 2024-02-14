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
import numpy as np
import pandas as pd

#1
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

columns = datasets.feature_names
x = pd.DataFrame(x, columns=columns)
y = pd.Series(y)
# 하위 20퍼센트의 인덱스: [3 4] 
to_delete = [3, 4]
for idx in sorted(to_delete, reverse=True):
    x = x.drop(x.columns[idx], axis=1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8)

#2
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

# DecisionTreeRegressor acc : 0.6072084013785619
# DecisionTreeRegressor feature importance
# [0.5138671  0.05258672 0.05090562 0.02825429 0.03213957 0.1353798
#  0.0941169  0.09275   ]
# 하위 20퍼센트의 인덱스: [3 4]

# RandomForestRegressor acc : 0.8189889822959849
# RandomForestRegressor feature importance
# [0.51891017 0.05349311 0.04663904 0.02986623 0.03206021 0.13609319
#  0.09198316 0.09095489]
# 하위 20퍼센트의 인덱스: [3 4] 

# GradientBoostingRegressor acc : 0.7930160644074155
# GradientBoostingRegressor feature importance
# [0.59787356 0.0300843  0.02454147 0.00436274 0.00281986 0.12187825
#  0.09591359 0.12252625]
# 하위 20퍼센트의 인덱스: [3 4] 

# XGBRegressor acc : 0.8386188269983407
# XGBRegressor feature importance
# [0.48721272 0.06797588 0.05002455 0.02223908 0.02304351 0.13496281
#  0.10358929 0.11095219]
# 하위 20퍼센트의 인덱스: [3 4] 
# #============================================================================
# DecisionTreeRegressor acc : 0.6270780840544892
# DecisionTreeRegressor feature importance
# [0.52513412 0.0581603  0.06579424 0.14223114 0.10520595 0.10347424]
# 하위 20퍼센트의 인덱스: [1]

# RandomForestRegressor acc : 0.8219260431435219
# RandomForestRegressor feature importance
# [0.52674621 0.05990473 0.05986584 0.14573763 0.10485464 0.10289094]
# 하위 20퍼센트의 인덱스: [2] 

# GradientBoostingRegressor acc : 0.793852357705537
# GradientBoostingRegressor feature importance
# [0.60079294 0.03235519 0.02447997 0.12272929 0.09778394 0.12185867]
# 하위 20퍼센트의 인덱스: [2]

# XGBRegressor acc : 0.8511323469254399
# XGBRegressor feature importance
# [0.51882464 0.06727694 0.05308819 0.14186715 0.10038856 0.1185545 ]
# 하위 20퍼센트의 인덱스: [2]