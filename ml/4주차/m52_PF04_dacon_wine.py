import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE    # anaconda에서 사이킷런 설치할 때 같이 설치됨    없다면  pip install imblearn
# from sklearn.impute import IterativeImputer 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

path  = 'C:\\_data\\dacon\\wine\\'

# 1. 데이터
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# print(y.value_counts())
lae = LabelEncoder()
train_csv['type'] = lae.fit_transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])

x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality'] - 3

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, stratify=y)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly_train = pf.fit_transform(x_train)
x_poly_test = pf.transform(x_test)

scaler = MinMaxScaler()
x_poly_train = scaler.fit_transform(x_poly_train)
x_poly_test = scaler.transform(x_poly_test)

# 2. 모델
# model = LogisticRegression()
# model = RandomForestClassifier()
model = XGBClassifier()
model2 = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)
model2.fit(x_poly_train, y_train)

# 4.
score = model.score(x_test, y_test)
class_name = model.__class__.__name__

score2 = model2.score(x_poly_test, y_test)
class_name = model.__class__.__name__

print("{0} ACC : {1:.6f}".format(class_name, score))
print("{0} ACC : {1:.6f}".format(class_name, score2))

# XGBClassifier ACC : 0.637091
# XGBClassifier ACC : 0.645091