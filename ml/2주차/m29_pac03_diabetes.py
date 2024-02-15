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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1226, train_size=0.9)

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

# (397, 1)
# model.score : 0.5130166159707578
# [0.39785721]
# (397, 2)
# model.score : 0.49713495772246397
# [0.39785721 0.54823128]
# (397, 3)
# model.score : 0.5531685872063137
# [0.39785721 0.54823128 0.67163549]
# (397, 4)
# model.score : 0.6253988058934279
# [0.39785721 0.54823128 0.67163549 0.76611893]
# (397, 5)
# model.score : 0.6623900108885626
# [0.39785721 0.54823128 0.67163549 0.76611893 0.8321986 ]
# (397, 6)
# model.score : 0.629408749472945
# [0.39785721 0.54823128 0.67163549 0.76611893 0.8321986  0.89303764]
# (397, 7)
# model.score : 0.6199258900884488
# [0.39785721 0.54823128 0.67163549 0.76611893 0.8321986  0.89303764
#  0.94799101]
# (397, 8)
# model.score : 0.6134141710586234
# [0.39785721 0.54823128 0.67163549 0.76611893 0.8321986  0.89303764
#  0.94799101 0.99133603]
# (397, 9)
# model.score : 0.6090900036045266
# [0.39785721 0.54823128 0.67163549 0.76611893 0.8321986  0.89303764
#  0.94799101 0.99133603 0.99918019]
# (397, 10)
# model.score : 0.6224622100461585
# [0.39785721 0.54823128 0.67163549 0.76611893 0.8321986  0.89303764
#  0.94799101 0.99133603 0.99918019 1.        ]

# 0.8321986 제일 좋음