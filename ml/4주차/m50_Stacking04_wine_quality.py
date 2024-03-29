import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
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

print(x.shape, y.shape)

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2 , random_state=777, stratify= y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
xgb = XGBClassifier(random_state=777)
rf = RandomForestClassifier(random_state=777)
lr = LogisticRegression(random_state=777, max_iter=1000)

model = StackingClassifier(
    estimators=[('XGB', xgb), ("RF", rf), ("LR", lr)],
    final_estimator=CatBoostClassifier(verbose=0),
    # stack_method='auto',  # 디폴트 'auto'로, 자동으로 적절한 쌓기 방법을 선택합니다. 'predict_proba' 또는 'predict' 중 선택할 수 있습니다.
    n_jobs=-1,
    cv=5
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
print(f"model.score : {model.score(x_test,y_test)}")
print("스태킹 ACC : ", accuracy_score(y_test, y_pred))

# model.score : 0.6745454545454546
# 스태킹 ACC :  0.6745454545454546

