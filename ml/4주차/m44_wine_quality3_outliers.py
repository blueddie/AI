import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score
# from sklearn.impute import IterativeImputer 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

path  = 'C:\\_data\\dacon\\wine\\'

# 1. 데이터
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# print(train_csv.shape, test_csv.shape)  #(5497, 13) (1000, 12)
# print(train_csv.dtypes)
# quality                   int64
# fixed acidity           float64
# volatile acidity        float64
# citric acid             float64
# residual sugar          float64
# chlorides               float64
# free sulfur dioxide     float64
# total sulfur dioxide    float64
# density                 float64
# pH                      float64
# sulphates               float64
# alcohol                 float64
# type                     object

x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality'] - 3

numeric_column_names = x.select_dtypes(include=['number']).columns.tolist()
# print(numeric_column_names)
# print(x.shape, y.shape) #(5497, 12) (5497,)
lae = LabelEncoder()
x['type'] = lae.fit_transform(x['type'])
test_csv['type'] = lae.transform(test_csv['type'])

# print(x['type'].value_counts())
# imputer = IterativeImputer()        # 선형회귀 알고리즘

def outlierHandler(data, labels):
    data = pd.DataFrame(data)
    
    for label in labels:
        series = data[label]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        
        iqr = q3 - q1
        upper_bound = q3 + (iqr * 1.3)
        lower_bound = q1 - (iqr * 1.3)
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        
        print(label, series.isna().sum())
        # series = series.interpolate()
        imputer = SimpleImputer(strategy='most_frequent')
        new_series = imputer.fit_transform(series.values.reshape(-1, 1))
        data[label] = new_series
        
        data = data.fillna(data.ffill())
        data = data.fillna(data.bfill())

    return data

out_x = outlierHandler(x, numeric_column_names)

# scaler = StandardScaler()
# out_x['citric acid'] = scaler.fit_transform(out_x['citric acid'].values.reshape(-1, 1))

# grouped_counts = out_x.groupby('citric acid').size()
# print(grouped_counts)
# # 시각화
# plt.bar(grouped_counts.index, grouped_counts.values)
# plt.xlabel('Label')
# plt.ylabel('Count')
# plt.title('Count of Labels')
# plt.show()




# fixed acidity 507
# volatile acidity 595
# citric acid 775
# residual sugar 355
# chlorides 435
# free sulfur dioxide 156
# total sulfur dioxide 67
# density 32
# pH 247
# sulphates 338
# alcohol 90
# type 1338
x_train, x_test, y_train, y_test = train_test_split(out_x, y, random_state=1234, train_size=0.8, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



parameters = {
    'max_depth': 9,  # 트리의 최대 깊이를 설정합니다.
    'learning_rate': 0.1,  # 학습률을 설정합니다.
    'n_estimators': 100,  # 트리의 개수를 설정합니다.
    'subsample': 0.8,  # 각 트리마다 사용될 샘플의 비율을 설정합니다.
    'colsample_bytree': 0.8,  # 각 트리마다 사용될 피처의 비율을 설정합니다.
    'reg_alpha': 0,  # L1 정규화 파라미터를 설정합니다.
    'reg_lambda': 1,  # L2 정규화 파라미터를 설정합니다.
    'random_state': 42  # 랜덤 시드를 설정합니다.
}

# 2. 모델
model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)

# 3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric='mlogloss'
          )

# 4. 평가, 예측
results = model.score(x_test, y_test)
# print("최종 점수 : ", results)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print(f"acc : {acc}")
