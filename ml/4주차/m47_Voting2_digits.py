import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
xgb = XGBClassifier(random_state=777)
rf = RandomForestClassifier(random_state=777)
lr = LogisticRegression(random_state=777)


model = BaggingClassifier(xgb,
                          n_estimators=10, # 디폴트
                          n_jobs=-2,
                          random_state=777,
                        #   bootstrap=True,   # 디폴트 중복을 허용한다
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

# bootstrap=False
# 최종 점수 :  0.9805555555555555
# acc_score :  0.9805555555555555

# bootstrap=True
# 최종 점수 :  0.9722222222222222
# acc_score :  0.9722222222222222