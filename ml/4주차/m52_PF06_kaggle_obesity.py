# https://www.kaggle.com/competitions/playground-series-s4e2/overview

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, RandomizedSearchCV, GridSearchCV
import time
from sklearn.metrics import accuracy_score
import warnings
from scipy.stats import uniform, randint
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings("ignore")

def outlierHandler(data, labels):
    data = pd.DataFrame(data)
    
    for label in labels:
        series = data[label]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        
        # print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
        data = data.fillna(data.ffill())
        data = data.fillna(data.bfill())

    return data

#1. 데이터
csv_path = 'C:\\_data\\kaggle\\obesity\\'

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

xy = train_csv.copy()
x_pred = test_csv.copy()

columns_to_drop = ['NObeyesdad']
x = xy.drop(columns=columns_to_drop)
y = xy[columns_to_drop]

non_float_x = []
numeric_x = []
for col in x.columns:
    if x[col].dtype != 'float64':
        non_float_x.append(col)
    else :
        numeric_x.append(col)
        
# print(numeric_x)
x = outlierHandler(x, numeric_x)
# print(pd.isna(x).sum())
# print(non_float)    #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
non_float_pred = []
numeric_test = []
for col in x_pred.columns:
    if x_pred[col].dtype != 'float64':
        non_float_pred.append(col)
    else :
        numeric_test.append(col)
# print(non_float_pred)   #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
print(numeric_test)
x_pred = outlierHandler(x_pred, numeric_test)


for col in non_float_x:
    print(f'x : {pd.value_counts(x[col])}')
    print(f'x_pred : {pd.value_counts(x_pred[col])}')
    print('------------------------------------')

# CALC -> Always 2 train에 없는 라벨 있음
x_pred['CALC'] = x_pred['CALC'].replace({'Always' : 'Sometimes'})
# print(pd.value_counts(x_pred['CALC']))

for column in x.columns:
    if (x[column].dtype != 'float64'):
        encoder = LabelEncoder()
        x[column] = encoder.fit_transform(x[column])
        x_pred[column] = encoder.transform(x_pred[column])
            
for col in x.columns :
    if x[col].dtype != 'float32':
        x[col] = x[col].astype('float32')
        x_pred[col] = x_pred[col].astype('float32')
# print(x.dtypes)
# print(x_pred.dtypes)


encoder = LabelEncoder()
y = encoder.fit_transform(y)

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

# XGBClassifier ACC : 0.888054
# XGBClassifier ACC : 0.886898