import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#1. 데이터
datasets = load_breast_cancer()

X = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=53, train_size=0.8, stratify=y)

#2 모델
model = RandomForestClassifier(random_state=777)

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
    
# (455, 1)
# model.score : 0.9035087719298246
# (455, 2)
# model.score : 0.9736842105263158
# (455, 3)
# model.score : 0.9385964912280702
# (455, 4)
# model.score : 0.9385964912280702
# (455, 5)
# model.score : 0.9649122807017544
# (455, 6)
# model.score : 0.956140350877193
# (455, 7)
# model.score : 0.956140350877193
# (455, 8)
# model.score : 0.956140350877193
# (455, 9)
# model.score : 0.956140350877193
# (455, 10)
# model.score : 0.9649122807017544
# (455, 11)
# model.score : 0.9473684210526315
# (455, 12)
# model.score : 0.9210526315789473
# (455, 13)
# model.score : 0.9473684210526315
# (455, 14)
# model.score : 0.9473684210526315
# (455, 15)
# model.score : 0.9473684210526315
# (455, 16)
# model.score : 0.956140350877193
# (455, 17)
# model.score : 0.9385964912280702
# (455, 18)
# model.score : 0.9385964912280702
# (455, 19)
# model.score : 0.9385964912280702
# (455, 20)
# model.score : 0.9473684210526315
# (455, 21)
# model.score : 0.9649122807017544
# (455, 22)
# model.score : 0.9473684210526315
# (455, 23)
# model.score : 0.9473684210526315
# (455, 24)
# model.score : 0.956140350877193
# (455, 25)
# model.score : 0.9298245614035088
# (455, 26)
# model.score : 0.9473684210526315
# (455, 27)
# model.score : 0.956140350877193
# (455, 28)
# model.score : 0.9473684210526315
# (455, 29)
# model.score : 0.9385964912280702
# (455, 30)
# model.score : 0.9385964912280702


# [0.44174598 0.63053193 0.72691376 0.79361544 0.84806713 0.88710849
#  0.91005614 0.92559929 0.94015686 0.95199046 0.96175652 0.97046033
#  0.97823339 0.98364648 0.98692678 0.98948957 0.99145073 0.99317793
#  0.99469004 0.99572061 0.99672627 0.99764626 0.99840899 0.998975
#  0.99945557 0.99971641 0.9999283  0.99997239 0.99999583 1.        ]

# 0.99672627 제일 좋음