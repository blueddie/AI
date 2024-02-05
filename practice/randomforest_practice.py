import xgboost as xgb
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE    # anacon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


csv_path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

xy = train_csv

unknown_replacement = xy['근로기간'].mode()[0]          # 최반값을 담았음
xy.loc[xy['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement       
test_csv.loc[test_csv['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement

# print(np.unique(xy['근로기간'], return_counts=True))
xy.loc[xy['근로기간'] == '<1 year', '근로기간'] = '< 1 year'
xy.loc[xy['근로기간'] == '3', '근로기간'] = '3 years'
xy.loc[xy['근로기간'] == '10+years', '근로기간'] = '10+ years'
xy.loc[xy['근로기간'] == '1 years', '근로기간'] = '1 year'


test_csv.loc[test_csv['근로기간'] == '<1 year', '근로기간'] = '< 1 year'
test_csv.loc[test_csv['근로기간'] == '3', '근로기간'] = '3 years'
test_csv.loc[test_csv['근로기간'] == '10+years', '근로기간'] = '10+ years'
test_csv.loc[test_csv['근로기간'] == '1 years', '근로기간'] = '1 year'

encoder = LabelEncoder()
encoder.fit(xy['근로기간'])
xy['근로기간'] = encoder.transform(xy['근로기간'])
test_csv['근로기간'] = encoder.transform(test_csv['근로기간'])

encoder = LabelEncoder()
encoder.fit(xy['대출기간'])
xy['대출기간'] = encoder.transform(xy['대출기간'])
test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])

xy.loc[xy['주택소유상태'] == 'ANY', '주택소유상태'] = 'MORTGAGE'

encoder.fit(xy['주택소유상태'])
xy['주택소유상태'] = encoder.transform(xy['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])

test_csv.loc[test_csv['대출목적'] == '결혼', '대출목적'] = '부채 통합'

encoder.fit(xy['대출목적'])
xy['대출목적'] = encoder.transform(xy['대출목적'])
test_csv['대출목적'] = encoder.transform(test_csv['대출목적'])

drop_column = ['대출등급']
x = xy.drop(columns=drop_column)
y = xy[drop_column]
# print(x.shape)  #(96294, 13)
y = y.values.reshape(-1)
print(y.shape)  #(96294, 1)

# x = x.astype(np.float32)
# y = x.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=555)
#2
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(x_train, y_train)


rf_model = RandomForestClassifier(n_estimators=256
                                  , max_depth=128
                                  , min_samples_split=2  #
                                  , min_samples_leaf=3
                                #   , max_features='auto'
                                  , bootstrap=True
)

################
#그리드서치
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='f1_macro')

################
grid_search.fit(x_train, y_train)
print("최적의 하이퍼파라미터:", grid_search.best_params_)

# rf_model.fit(x_train, y_train)
# y_pred = rf_model.predict(x_test)
y_pred = grid_search.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'모델 정확도: {accuracy}')
f1 = f1_score(y_test, y_pred, average='macro')
print(f'모델의 F1 점수: {f1}')
print(y_pred)

# 11, 37 2  모델의 F1 점수: 0.7509280751496095
# 32 37, 2 모델의 F1 점수: 0.7798120155563982
#128 37, 2  모델의 F1 점수: 0.7963095929316519
#256 37, 2  모델의 F1 점수: 0.8005862486504418
#256, 128 , 2 모델의 F1 점수: 0.8027732192023892


#macro
#256, 256 , 2 모델의 F1 점수: 0.6660656689245364




