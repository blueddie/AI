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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#1. 데이터
path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

X = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
test_csv = test_csv.drop([], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=15)
#2. 모델
#2 모델
model = RandomForestRegressor(random_state=777)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

for i in range(1, x_train.shape[1] + 1):
    
    pca = PCA(n_components=i)   
    x1 = pca.fit_transform(x_train)
    x2 = pca.transform(x_test)

    # #3.
    model.fit(x1, y_train)

    # #4.
    results = model.score(x2, y_test)
    print(x1.shape)
    print(f"model.score : {results}")
    evr = pca.explained_variance_ratio_
    evr_cumsum = np.cumsum(evr)
    print(evr_cumsum)

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



# (8708, 1)
# model.score : -0.1209030717786932
# [0.26500185]
# (8708, 2)
# model.score : 0.185257518898429
# [0.26500185 0.46241536]
# (8708, 3)
# model.score : 0.2862850550014241
# [0.26500185 0.46241536 0.61917112]
# (8708, 4)
# model.score : 0.31053327015020604
# [0.26500185 0.46241536 0.61917112 0.74743741]
# (8708, 5)
# model.score : 0.3153068399596187
# [0.26500185 0.46241536 0.61917112 0.74743741 0.84684314]
# (8708, 6)
# model.score : 0.31023416557419026
# [0.26500185 0.46241536 0.61917112 0.74743741 0.84684314 0.94039009]
# (8708, 7)
# model.score : 0.32322141697732265
# [0.26500185 0.46241536 0.61917112 0.74743741 0.84684314 0.94039009
#  0.99839974]
# (8708, 8)
# model.score : 0.3283254448100278
# [0.26500185 0.46241536 0.61917112 0.74743741 0.84684314 0.94039009            # 제일 좋음
#  0.99839974 1.        ]