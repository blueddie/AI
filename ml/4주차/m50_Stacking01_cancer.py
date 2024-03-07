import numpy as np
import tensorflow as tf
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
np.random.seed(777)
tf.random.set_seed(777)
random.seed(777)

import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
xgb = XGBClassifier(random_state=777)
rf = RandomForestClassifier(random_state=777)
lr = LogisticRegression(random_state=777)

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