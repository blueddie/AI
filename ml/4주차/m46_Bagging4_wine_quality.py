import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, f1_score
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

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, train_size=0.8, stratify=y)

# 2. 모델
model = BaggingClassifier(LogisticRegression(max_iter=2000), # max_iter DEFAULT 100
                          n_estimators=10, # 디폴트
                          n_jobs=-2,
                          random_state=777,
                          # bootstrap=True,   # 디폴트 중복을 허용한다
                          bootstrap=False # 중복 허용 X
                                                      
                          )

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print("최종 점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)


# bootstrap=True
# 최종 점수 :  0.5345454545454545
# acc_score :  0.5345454545454545

# bootstrap=False
# 최종 점수 :  0.5345454545454545
# acc_score :  0.5345454545454545