# https://dacon.io/competitions/open/236070/overview/description
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#1.
path = "C://_data//dacon//iris//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

x = train_csv.drop(['species'], axis=1)
y = train_csv['species']

# print(X.shape)  #(120, 4)
# print(y.shape)  #(120,)
print(pd.value_counts(y))



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=13, train_size=0.8, stratify=y)

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

# (96, 1)
# model.score : 0.9166666666666666
# [0.73127212]
# (96, 2)
# model.score : 0.875
# [0.73127212 0.95807857]
# (96, 3)
# model.score : 0.9583333333333334
# [0.73127212 0.95807857 0.99495539]
# (96, 4)
# model.score : 1.0
# [0.73127212 0.95807857 0.99495539 1.        ]