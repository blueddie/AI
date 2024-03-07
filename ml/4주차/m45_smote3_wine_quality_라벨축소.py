import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, f1_score
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
y = train_csv['quality']
# print(x.shape, y.shape)
print(y.value_counts())
def remove_outlier(dataset:pd.DataFrame):
    for label in dataset:
        data = dataset[label]
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3-q1
        upbound    = q3 + iqr*1.5
        underbound = q1 - iqr*1.5
        dataset.loc[dataset[label] < underbound, label] = underbound
        dataset.loc[dataset[label] > upbound, label] = upbound
        
    return dataset

x = x.astype(np.float32)
y = y.astype(np.float32)

y = y.copy()  # 알아서 참고

for i, v in enumerate(y):
    if v <= 4:
        y[i] = 0
    elif v == 5:
        y[i]=1
    elif v == 6:
        y[i]=2
    # elif v==7:
    #     y[i]=3    
    # elif v==8:
    #     y[i]=4
    else:
        y[i]=2
        
print(y.value_counts())



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, train_size=0.8, stratify=y)

smote = SMOTE(random_state=123, k_neighbors=3)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(pd.value_counts(y_train))

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
f1 = f1_score(y_test, y_predict, average='macro')
print(f"f1 : {f1}")

# 증폭 전
# acc : 0.8009090909090909
# f1 : 0.6145012728409012

# 증폭 후
# acc : 0.7936363636363636
# f1 : 0.6596686668906738