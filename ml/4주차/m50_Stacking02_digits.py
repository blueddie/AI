import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
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

# bootstrap=False
# 최종 점수 :  0.9805555555555555
# acc_score :  0.9805555555555555

# bootstrap=True
# 최종 점수 :  0.9722222222222222
# acc_score :  0.9722222222222222

# model.score : 0.9777777777777777
# 스태킹 ACC :  0.9777777777777777