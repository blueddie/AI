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

# DataFrame
columns = datasets.feature_names
x = pd.DataFrame(x, columns=columns)
y = pd.Series(y)
print(x.shape)

# 하위 20퍼센트의 인덱스: [1 3 8]
to_delete = [1, 3 , 8]
for idx in sorted(to_delete, reverse=True):
    x = x.drop(x.columns[idx], axis=1)
print(x.shape)




x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8)

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

# DecisionTreeRegressor acc : 0.8124734162242866
# DecisionTreeRegressor feature importance
# [2.32295509e-02 3.38779186e-04 1.00871727e-02 9.00388502e-04
#  2.98213256e-02 2.43897337e-01 1.25072843e-02 9.03304062e-02
#  4.45281926e-03 9.54738406e-03 2.82642283e-02 9.51216545e-03
#  5.37111159e-01]
# 하위 20퍼센트의 인덱스: [1 3 8]

# RandomForestRegressor acc : 0.9132156681256265
# RandomForestRegressor feature importance
# [0.03446112 0.00095869 0.00624401 0.00149858 0.02994178 0.34611351
#  0.01634286 0.08129735 0.00374252 0.01367285 0.01387266 0.01059616
#  0.44125791]
# 하위 20퍼센트의 인덱스: [1 3 8]

# GradientBoostingRegressor acc : 0.9288073876145365
# GradientBoostingRegressor feature importance
# [1.57715910e-02 2.57123457e-04 3.37686459e-03 1.20748480e-03
#  4.04700503e-02 2.98435368e-01 7.20264442e-03 1.02088251e-01
#  2.16057511e-03 1.46830508e-02 3.56318919e-02 8.39498843e-03
#  4.70320116e-01]
# 하위 20퍼센트의 인덱스: [1 3 8]

# XGBRegressor acc : 0.9022178792512646
# XGBRegressor feature importance
# [0.01257844 0.00061468 0.01847021 0.00281567 0.04047003 0.1499025
#  0.00947146 0.10018896 0.01814521 0.03351607 0.02434715 0.00769892
#  0.58178073]
# 하위 20퍼센트의 인덱스: [ 1  3 11]

#==========================================================================
# DecisionTreeRegressor acc : 0.827475211211695
# DecisionTreeRegressor feature importance
# [0.02313194 0.00798378 0.031102   0.24416881 0.01168692 0.09098254
#  0.00997459 0.03074165 0.01088104 0.53934673]
# 하위 20퍼센트의 인덱스: [1 6]

# RandomForestRegressor acc : 0.9133188817309168
# RandomForestRegressor feature importance
# [0.03743329 0.00722944 0.03052201 0.34606332 0.01740742 0.07979097
#  0.01532979 0.01461601 0.01080621 0.44080155]
# 하위 20퍼센트의 인덱스: [1 8]

# GradientBoostingRegressor acc : 0.9352290500808976
# GradientBoostingRegressor feature importance
# [0.01612125 0.00344937 0.04259734 0.30197249 0.00714558 0.10176915
#  0.01489187 0.03363209 0.00795247 0.4704684 ]
# 하위 20퍼센트의 인덱스: [1 4]

# XGBRegressor acc : 0.9059812475542829
# XGBRegressor feature importance
# [0.01462013 0.0197483  0.03485354 0.17740497 0.01042872 0.09941243
#  0.03231266 0.03491132 0.00729357 0.5690144 ]
# 하위 20퍼센트의 인덱스: [4 8]