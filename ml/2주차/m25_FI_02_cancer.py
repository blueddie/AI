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


#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# DataFrame
columns = datasets.feature_names
x = pd.DataFrame(x, columns=columns)
y = pd.Series(y)
print(x.shape)
# 하위 20퍼센트의 인덱스: [ 2  8 12 14 18 25] 
to_delete = [2, 8, 12, 14, 18, 25]
for idx in sorted(to_delete, reverse=True):
    x = x.drop(x.columns[idx], axis=1)
print(x.shape)



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=53, train_size=0.8, stratify=y)

#2 모델
# model = LinearSVC(C=100)        # C가 크면 training포인트를 정확히 구분(굴곡지다), C가 작으면 직선에 가깝다.
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

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


# 하위 20퍼센트의 인덱스: [ 2  8 12 14 18 25] 
    
    
    
    
    
# DecisionTreeClassifier acc : 0.9122807017543859
# DecisionTreeClassifier feature importance
# [0.         0.04299701 0.00626075 0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.04125562 0.0022404  0.         0.01760836 0.
#  0.         0.01408669 0.         0.03742135 0.69925315 0.01576672
#  0.00932356 0.01811146 0.         0.09567494 0.         0.        ]

# RandomForestClassifier acc : 0.9649122807017544
# RandomForestClassifier feature importance
# [0.02983656 0.01183896 0.04076089 0.05106236 0.00569148 0.00523903
#  0.04455282 0.10714978 0.00441927 0.00274717 0.00920409 0.00501033
#  0.01004834 0.02765508 0.00377432 0.00334234 0.00432161 0.00416023
#  0.00358639 0.00599841 0.12419539 0.01565628 0.22828686 0.10014622
#  0.01829904 0.02150172 0.01886845 0.07398958 0.0082709  0.0103861 ]

# GradientBoostingClassifier acc : 0.9736842105263158
# GradientBoostingClassifier feature importance
# [1.31939689e-03 2.16614680e-02 3.49963155e-04 8.98496501e-03
#  4.91159101e-03 1.52045028e-03 5.44231906e-03 3.11837292e-02
#  5.47396883e-04 6.78143989e-07 4.37814891e-03 9.26779047e-03
#  8.34925960e-04 4.67776849e-03 1.60619959e-03 1.34095412e-03
#  1.36740100e-03 3.09107940e-03 0.00000000e+00 6.05753186e-04
#  2.42436507e-01 4.84420633e-02 4.56792722e-01 2.98629691e-02
#  1.99855514e-02 3.57992756e-04 3.46457647e-03 9.25276067e-02
#  1.43029692e-03 1.60773605e-03]

# XGBClassifier acc : 0.9824561403508771
# XGBClassifier feature importance
# [2.1164339e-02 1.8960079e-02 0.0000000e+00 3.4322668e-02 4.9383924e-03
#  2.8294218e-03 7.7569075e-03 8.0713026e-02 1.2768973e-03 4.1659730e-03
#  6.6776555e-03 2.4561500e-03 2.1841312e-03 5.6698131e-03 2.1871186e-03
#  4.1694804e-03 7.2960863e-03 2.5183205e-02 1.3305614e-03 2.8682374e-03
#  2.7113535e-02 1.8029649e-02 5.4660314e-01 2.5041355e-02 8.8974154e-03
#  4.0043134e-04 1.9765066e-02 1.0810049e-01 3.0872624e-03 6.8114665e-03]

#================================================================================
# DecisionTreeClassifier acc : 0.9210526315789473
# DecisionTreeClassifier feature importance
# [0.         0.03595367 0.         0.         0.         0.
#  0.         0.         0.00914427 0.         0.0383721  0.
#  0.01760836 0.         0.0009368  0.01173891 0.03742063 0.70629649
#  0.01576672 0.00932356 0.         0.1174385  0.         0.        ]
# 하위 20퍼센트의 인덱스: []

# RandomForestClassifier acc : 0.956140350877193
# RandomForestClassifier feature importance
# [0.04644477 0.01552481 0.06141809 0.00840394 0.0142043  0.0806799
#  0.08335529 0.00556755 0.02703527 0.00607733 0.04298587 0.00591759
#  0.01855075 0.00992476 0.00576453 0.13151652 0.01811958 0.11242515
#  0.12776292 0.01611989 0.05894737 0.08023394 0.01461307 0.00840683]
# 하위 20퍼센트의 인덱스: [ 3  7  9 11 14] 

# GradientBoostingClassifier acc : 0.9824561403508771
# GradientBoostingClassifier feature importance
# [6.18374554e-04 2.04693498e-02 7.21822064e-03 3.66556071e-03
#  3.26221375e-03 9.90787864e-03 3.20218048e-02 2.53739414e-04
#  5.24955480e-03 9.28006268e-03 6.90498630e-03 9.50487199e-04
#  4.46284901e-03 3.09578110e-03 8.47329839e-04 2.41101449e-01
#  5.00990866e-02 4.55463232e-01 3.10640582e-02 1.72107170e-02
#  5.24244199e-03 8.85221370e-02 2.98857677e-03 1.00107920e-04]
# 하위 20퍼센트의 인덱스: [ 0  7 11 14 23]

# XGBClassifier acc : 0.9912280701754386
# XGBClassifier feature importance
# [0.01967609 0.02179894 0.01934014 0.00725873 0.00201618 0.00240969
#  0.17923327 0.00659255 0.01129503 0.00315691 0.00645894 0.00448545
#  0.00419658 0.0112626  0.00227375 0.04195111 0.02227131 0.50310785
#  0.02432987 0.01032342 0.02248616 0.06339371 0.00312023 0.00756147]
# 하위 20퍼센트의 인덱스: [ 4  5  9 14 22]