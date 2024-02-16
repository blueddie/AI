from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1.
datasets = load_wine()

X = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=13, train_size=0.8, stratify=y)

#2 모델
model = RandomForestClassifier(random_state=777)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_features = len(np.unique(y))

for i in range(1, n_features):
    
    lda = LinearDiscriminantAnalysis(n_components=i)   
    x1 = lda.fit_transform(x_train, y_train)
    x2 = lda.transform(x_test)

    # #3.
    model.fit(x1, y_train)

    # #4.
    results = model.score(x2, y_test)
    print(x1.shape)
    print(f"model.score : {results}")
    evr = lda.explained_variance_ratio_
    evr_cumsum = np.cumsum(evr)
    print(evr_cumsum)
    
# (142, 1)
# model.score : 0.7777777777777778
# (142, 2)
# model.score : 0.9166666666666666
# (142, 3)
# model.score : 0.9444444444444444
# (142, 4)
# model.score : 0.9166666666666666
# (142, 5)
# model.score : 0.9166666666666666
# (142, 6)
# model.score : 0.9166666666666666
# (142, 7)
# model.score : 0.9444444444444444
# (142, 8)
# model.score : 0.9722222222222222
# (142, 9)
# model.score : 0.9444444444444444
# (142, 10)
# model.score : 0.9444444444444444
# (142, 11)
# model.score : 0.9444444444444444
# (142, 12)
# model.score : 0.9444444444444444
# (142, 13)
# model.score : 0.9444444444444444

#
# (142, 1)
# model.score : 0.9166666666666666
# [0.72153931]
# (142, 2)
# model.score : 0.9722222222222222
# [0.72153931 1.        ]