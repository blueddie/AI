import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


csv_path = 'C:\\_data\\kaggle\\obesity\\'

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

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
# print(non_float_pred)   #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

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

sns.set(font_scale=0.8)
sns.heatmap(data=x.corr(), square=True, annot=True, cbar=True)
plt.show()