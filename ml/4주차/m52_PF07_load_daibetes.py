import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly_train = pf.fit_transform(x_train)
x_poly_test = pf.transform(x_test)

scaler = MinMaxScaler()
x_poly_train = scaler.fit_transform(x_poly_train)
x_poly_test = scaler.transform(x_poly_test)

# 2. 모델
# model = LogisticRegression()
# model = RandomForestClassifier()
model = XGBRegressor()
model2 = XGBRegressor()

# 3. 훈련
model.fit(x_train, y_train)
model2.fit(x_poly_train, y_train)

# 4.
score = model.score(x_test, y_test)
class_name = model.__class__.__name__

score2 = model2.score(x_poly_test, y_test)
class_name = model.__class__.__name__

print("{0} r2 : {1:.6f}".format(class_name, score))
print("{0} r2 : {1:.6f}".format(class_name, score2))

# XGBRegressor r2 : 0.163780
# XGBRegressor r2 : 0.065931