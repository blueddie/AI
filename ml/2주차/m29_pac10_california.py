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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
#1
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8)

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

# (16512, 1)
# model.score : -0.27135423371394496
# [0.25189445]
# (16512, 2)
# model.score : 0.18325811071096032
# [0.25189445 0.48746192]
# (16512, 3)
# model.score : 0.3542490385252429
# [0.25189445 0.48746192 0.64630702]
# (16512, 4)
# model.score : 0.5532932987029229
# [0.25189445 0.48746192 0.64630702 0.77604261]
# (16512, 5)
# model.score : 0.5871647746648014
# [0.25189445 0.48746192 0.64630702 0.77604261 0.90138771]
# (16512, 6)
# model.score : 0.6508268900666746
# [0.25189445 0.48746192 0.64630702 0.77604261 0.90138771 0.98331714]
# (16512, 7)
# model.score : 0.7156404137012943
# [0.25189445 0.48746192 0.64630702 0.77604261 0.90138771 0.98331714
#  0.99399724]
# (16512, 8)
# model.score : 0.7332490651644745
# [0.25189445 0.48746192 0.64630702 0.77604261 0.90138771 0.98331714
#  0.99399724 1.        ]