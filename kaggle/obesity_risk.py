# https://www.kaggle.com/competitions/playground-series-s4e2/overview

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm  as lgb
import catboost as cb
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

csv_path = 'C:\\_data\\kaggle\\obesity\\'

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

# print(train_csv.shape)    #[20758 rows x 17 columns]    (20758, 17)     
# print(test_csv.shape)   #[13840 rows x 16 columns]        (13840, 16)      

# print(train_csv.describe())
#                 Age        Height        Weight          FCVC           NCP          CH2O           FAF           TUE
# count  20758.000000  20758.000000  20758.000000  20758.000000  20758.000000  20758.000000  20758.000000  20758.000000
# mean      23.841804      1.700245     87.887768      2.445908      2.761332      2.029418      0.981747      0.616756
# std        5.688072      0.087312     26.379443      0.533218      0.705375      0.608467      0.838302      0.602113
# min       14.000000      1.450000     39.000000      1.000000      1.000000      1.000000      0.000000      0.000000
# 25%       20.000000      1.631856     66.000000      2.000000      3.000000      1.792022      0.008013      0.000000
# 50%       22.815416      1.700000     84.064875      2.393837      3.000000      2.000000      1.000000      0.573887
# 75%       26.000000      1.762887    111.600553      3.000000      3.000000      2.549617      1.587406      1.000000
# max       61.000000      1.975663    165.057269      3.000000      4.000000      3.000000      3.000000      2.000000

# print(test_csv.describe())
#                 Age        Height        Weight          FCVC           NCP          CH2O           FAF           TUE
# count  13840.000000  13840.000000  13840.000000  13840.000000  13840.000000  13840.000000  13840.000000  13840.000000
# mean      23.952740      1.698934     87.384504      2.442898      2.750610      2.032044      0.974532      0.611033
# std        5.799814      0.088761     26.111819      0.531606      0.710927      0.611230      0.840361      0.608005
# min       14.000000      1.450000     39.000000      1.000000      1.000000      1.000000      0.000000      0.000000
# 25%       20.000000      1.631662     65.000000      2.000000      3.000000      1.771781      0.001086      0.000000
# 50%       22.906342      1.700000     83.952968      2.358087      3.000000      2.000000      1.000000      0.552498
# 75%       26.000000      1.760710    111.157811      3.000000      3.000000      2.552388      1.571865      1.000000
# max       61.000000      1.980000    165.057269      3.000000      4.000000      3.000000      3.000000      2.000000

# print(pd.isna(train_csv).sum()) # 결측치 없음
# print(pd.isna(test_csv).sum()) # 결측치 없음
# 1.데이터
xy = train_csv.copy()
x_pred = test_csv.copy()

columns_to_drop = ['NObeyesdad']
x = xy.drop(columns=columns_to_drop)
y = xy[columns_to_drop]
# print(x.shape)  #(20758, 16)
# print(y.shape)    #(20758, 1)
non_float_x = []
for col in x.columns:
    if x[col].dtype != 'float64':
        non_float_x.append(col)
# print(non_float)    #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
non_float_pred = []
for col in x_pred.columns:
    if x_pred[col].dtype != 'float64':
        non_float_pred.append(col)
print(non_float_pred)   #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

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


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=337, stratify=y)
#2 모델

# model = RandomForestClassifier()
# model = lgb.LGBMClassifier()
# model = cb.CatBoost()
model = xgb.XGBClassifier()

# #3.
model.fit(x_train, y_train)

# #4.
# from sklearn.metrics import accuracy_score
results = model.score(x_test, y_test)
# results = accuracy_score(y_test, y_pred)
y_pred = model.predict(x_test)

y_submit = model.predict(x_pred)
y_submit = encoder.inverse_transform(y_submit)

# print('acc : ' , results)

import datetime

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   

submission_csv['NObeyesdad'] = pd.DataFrame(y_submit.reshape(-1,1))
submission_csv.to_csv(csv_path + f"{date}_{model.__class__.__name__}_acc_{results:.2f}.csv", index=False)

