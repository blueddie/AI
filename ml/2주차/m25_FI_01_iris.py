from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)


class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return "XGBClassifier()"


#1 . 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(datasets.feature_names)
columns = datasets.feature_names
print(x.shape, y.shape)     # (150, 4) (150,)

# 실습
#  피쳐 임포턴스가 전체 중요도에서 하위 20-25% 컬럼들을 제거하여
# 데이터셋 재구성

# numpy 컬럼 삭제
# x = np.delete(x, 0, axis=1)

# 판다스 데이터프레임 삭제
x = pd.DataFrame(x, columns=columns)
y = pd.Series(y)
x = x.drop(x.columns[0], axis=1)
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=13, train_size=0.8, stratify=y)

#2 모델
# model = LinearSVC(C=100)        # C가 크면 training포인트를 정확히 구분(굴곡지다), C가 작으면 직선에 가깝다.
# model = DecisionTreeClassifier(random_state=777)
# model = RandomForestClassifier(random_state=777)
# model = GradientBoostingClassifier(random_state=777)
# model = XGBClassifier(random_state=777)

models = [DecisionTreeClassifier(random_state=777), RandomForestClassifier(random_state=777), GradientBoostingClassifier(random_state=777), CustomXGBClassifier(random_state=777)]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(model, "acc :", results)  

    print(f"{model} feature importance\n{model.feature_importances_}")


   # [0.01666667 0.01666667 0.38920455 0.57746212] 컬럼의 중요도? 첫번째 컬럼은 중요도가 낮다.


# DecisionTreeClassifier acc : 1.0
# DecisionTreeClassifier feature importance
# [0.01666667 0.01666667 0.88920455 0.07746212]

# RandomForestClassifier acc : 1.0
# RandomForestClassifier feature importance
# [0.10654043 0.0317527  0.4098676  0.45183928]

# GradientBoostingClassifier acc : 1.0
# GradientBoostingClassifier feature importance
# [0.00460717 0.02041063 0.52575883 0.44922336]

# XGBClassifier acc : 1.0
# XGBClassifier feature importance
# [0.01490525 0.02423168 0.7842199  0.17664316]
