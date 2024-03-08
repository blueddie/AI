# https://dacon.io/competitions/official/236214/mysubmission
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')


csv_path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

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
        
        print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
        data = data.fillna(data.ffill())
        data = data.fillna(data.bfill())

    return data

# print(train_csv.shape, test_csv.shape)  #(96294, 14) (64197, 13)
# print(train_csv.columns)
# Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수', '대출등급'],
#       dtype='object')
# print(test_csv.columns)
# Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수'],
#       dtype='object')

# 결측치 확인
# print(pd.isna(train_csv).sum()) # 결측치 없음
# print(pd.isna(test_csv).sum()) # 결측치 없음

# print(train_csv.dtypes)
# print(test_csv.dtypes)

x = train_csv.drop(['대출등급'], axis=1)
y  = train_csv['대출등급']

# print(x.shape)  #(96294, 13)

non_numeric_train = []
numeric_train = []

for label in x.columns:
    if x[label].dtype not in ['float64', 'int64']:
        non_numeric_train.append(label)
    else : 
        numeric_train.append(label)
        
# print(non_numeric_train)  # ['대출기간', '근로기간', '주택소유상태', '대출목적']
# print(numeric_train)      #['대출금액', '연간소득', '부채_대비_소득_비율', '총계좌수', '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수']

x = outlierHandler(x, numeric_train)     

non_numeric_test = []
numeric_test = []
for label in x.columns:
    if test_csv[label].dtype not in ['float64', 'int64']:
        non_numeric_test.append(label)
    else :
        numeric_test.append(label)        
# print(non_numeric_test)
# ['대출기간', '근로기간', '주택소유상태', '대출목적']

test_csv = outlierHandler(test_csv, numeric_test)

# for label in non_numeric:
#     print(f'train_csv : {pd.value_counts(x[label])}')
#     print(f'test_csv : {pd.value_counts(test_csv[label])}')
#     print('------------------------------------')

x.loc[x['주택소유상태'] == 'ANY', '주택소유상태'] = 'MORTGAGE'
# print(np.unique(x['주택소유상태']))
test_csv.loc[test_csv['대출목적'] == '결혼' ,'대출목적'] = '기타'

# print(np.unique(test_csv['대출목적']))

unknown_replacement = x['근로기간'].mode()[0]
x.loc[x['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement
test_csv.loc[test_csv['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement


x.loc[x['근로기간'] == '<1 year', '근로기간'] = '< 1 year'
x.loc[x['근로기간'] == '3', '근로기간'] = '3 years'
x.loc[x['근로기간'] == '10+years', '근로기간'] = '10+ years'
x.loc[x['근로기간'] == '1 years', '근로기간'] = '1 year'

test_csv.loc[test_csv['근로기간'] == '<1 year', '근로기간'] = '< 1 year'
test_csv.loc[test_csv['근로기간'] == '3', '근로기간'] = '3 years'
test_csv.loc[test_csv['근로기간'] == '10+years', '근로기간'] = '10+ years'
test_csv.loc[test_csv['근로기간'] == '1 years', '근로기간'] = '1 year'

for label in non_numeric_train:
    lae = LabelEncoder()
    x[label] = lae.fit_transform(x[label])
    test_csv[label] = lae.fit_transform(test_csv[label])

x = x.astype('float32')
test_csv = test_csv.astype('float32')

lae = LabelEncoder()
y = lae.fit_transform(y)

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

# XGBClassifier ACC : 0.781050
# XGBClassifier ACC : 0.778225