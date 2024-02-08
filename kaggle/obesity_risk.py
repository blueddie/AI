# https://www.kaggle.com/competitions/playground-series-s4e2/overview

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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





# for col in x.columns :
#     if x[col].dtype == 'float64':
#         x[col] = x[col].astype('float32')
# print(x.dtypes)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=337, stratify=y)
#2 모델

# model = RandomForestClassifier()

# #3.
# model.fit(x_train, y_train)

# #4.
# results = model.score(x_test, y_test)
# print('acc : ' , results)