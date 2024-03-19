# https://www.kaggle.com/competitions/playground-series-s4e2/overview

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, RandomizedSearchCV, GridSearchCV
import time
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings

warnings.filterwarnings("ignore")

#1. 데이터
csv_path = 'C:\\_data\\kaggle\\obesity\\'

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

df = train_csv.copy()
x_pred = test_csv.copy()

non_float_x = []
for col in df.columns:
    if df[col].dtype != 'float64':
        non_float_x.append(col)
# print(non_float)    #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
non_float_pred = []
for col in x_pred.columns:
    if x_pred[col].dtype != 'float64':
        non_float_pred.append(col)
# print(non_float_pred)   #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

for col in non_float_x:
    print(f'df : {pd.value_counts(df[col])}')
    print(f'x_pred : {pd.value_counts(x_pred[col])}')
    print('------------------------------------')

x_pred['CALC'] = x_pred['CALC'].replace({'Always' : 'Sometimes'})
# x_pred['CALC'] = x_pred['CALC'].replace({'Always' : 'Sometimes'})

for column in df.columns:
    if (df[column].dtype != 'float64'):
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        x_pred[column] = encoder.transform(x_pred[column])
            
for col in df.columns :
    if df[col].dtype != 'float32':
        df[col] = df[col].astype('float32')
        x_pred[col] = x_pred[col].astype('float32')
        
print(df)