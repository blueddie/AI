import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()
# rf2 = RandomForestClassifier()

models = [xgb, rf, lr]
li = []
li2 = []

for model in models:
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    # print(y_pred.shape)         #(455,)
    
    li.append(y_pred)
    li2.append(y_pred_test)
    
    score = accuracy_score(y_test, y_pred_test)
    class_name = model.__class__.__name__
    print("{0} ACC : {1:.4f}".format(class_name, score))
    
    
new_x_train = np.array(li).T
new_x_test = np.array(li2).T

print(new_x_train.shape, new_x_test.shape)  #(455, 3) (114, 3)
# new_features = np.column_stack(li)
# print(new_features)
# print(new_features.shape)   #(114, 3)

# new_features2 = np.vstack(predictions).T
# print(new_features2)
# print(new_features2.shape)  ##(114, 3)
# print(new_features2)

# new_features3 = np.array(predictions).T
# print(new_features2)
# print(new_features3.shape)  ##(114, 3)
# print(new_features2)

model2 = CatBoostClassifier()
model2.fit(new_x_train, y_train, verbose=0)
y_pred = model2.predict(new_x_test)

score2 = accuracy_score(y_test, y_pred)
class_name = model2.__class__.__name__
print("{0} ACC : {1:.6f}".format(class_name, score2))
