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