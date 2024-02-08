import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import pandas as pd

#1. 데이터
# x, y = load_iris(return_X_y=True)
datasets = load_iris()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# print(df)

n_splits = 3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

for train_index, val_index in kfold.split(df):
    print(f'{train_index}\n{val_index}')
    print(f'훈련 데이터 갯수 : {len(train_index)}\n검증 데이터 갯수 {len(val_index)}')
    print("===========================")

'''
#2. 모델
model = SVC()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)  #cv 교차검증
# print('ACC : ', scores)
print(f'ACC : {scores}\n평균 ACC: {round(np.mean(scores), 4)}')

# 평균 ACC: 0.9667
'''