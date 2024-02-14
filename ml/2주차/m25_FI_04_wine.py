from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
#1.
datasets = load_wine()

X = datasets.data
y = datasets.target

columns = datasets.feature_names
X = pd.DataFrame(X, columns=columns)
y = pd.Series(y)
print(X.shape)
# 하위 20퍼센트의 인덱스: [2 7 8]
to_delete = [2, 7, 8]
for idx in sorted(to_delete, reverse=True):
    X = X.drop(X.columns[idx], axis=1)
print(X.shape)



x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=13, train_size=0.8, stratify=y)

models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(model.__class__.__name__, "acc :", results)  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")
    threshold = np.percentile(model.feature_importances_, 20)  
    low_importance_indices = np.where(model.feature_importances_ < threshold)[0]   
    print("하위 20퍼센트의 인덱스:", low_importance_indices, "\n")

    
# DecisionTreeClassifier acc : 0.9166666666666666
# DecisionTreeClassifier feature importance
# [0.         0.         0.02086548 0.         0.         0.
#  0.40558571 0.         0.         0.38697541 0.         0.04111597
#  0.14545744]
# 하위 20퍼센트의 인덱스: []

# RandomForestClassifier acc : 1.0
# RandomForestClassifier feature importance
# [0.13470934 0.01984028 0.01521587 0.03365348 0.02055278 0.05938374
#  0.15518301 0.01098936 0.01549318 0.17631641 0.07153648 0.11483398
#  0.1722921 ]
# 하위 20퍼센트의 인덱스: [2 7 8]

# GradientBoostingClassifier acc : 0.9722222222222222
# GradientBoostingClassifier feature importance
# [3.34253680e-03 5.84209977e-02 7.09844579e-03 4.46660432e-03
#  9.62725741e-03 2.05222232e-03 1.91815528e-01 4.26268000e-04
#  1.79367057e-05 2.99588783e-01 2.16129196e-02 1.35818695e-01
#  2.65711805e-01]
# 하위 20퍼센트의 인덱스: [5 7 8]

# XGBClassifier acc : 0.9444444444444444
# XGBClassifier feature importance
# [0.01018853 0.05218591 0.0091292  0.00567487 0.01885441 0.0277835
#  0.14115812 0.01073536 0.00816022 0.18099341 0.04160717 0.3505479
#  0.14298138]
# 하위 20퍼센트의 인덱스: [2 3 8]

#==========================================================================
# DecisionTreeClassifier acc : 0.8888888888888888
# DecisionTreeClassifier feature importance
# [0.         0.         0.         0.         0.         0.40558571
#  0.38697541 0.02086548 0.04111597 0.14545744]
# 하위 20퍼센트의 인덱스: []

# RandomForestClassifier acc : 1.0
# RandomForestClassifier feature importance
# [0.12282142 0.02656004 0.02154268 0.04148198 0.06432302 0.16625959
#  0.18296186 0.08121285 0.10785704 0.18497951]
# 하위 20퍼센트의 인덱스: [1 2]

# GradientBoostingClassifier acc : 0.9722222222222222
# GradientBoostingClassifier feature importance
# [0.00189805 0.05330118 0.00261929 0.01907504 0.00294292 0.19389256
#  0.29985298 0.02518762 0.13361555 0.26761481]
# 하위 20퍼센트의 인덱스: [0 2]

# XGBClassifier acc : 0.9722222222222222
# XGBClassifier feature importance
# [0.01081002 0.05091118 0.00652879 0.01256187 0.03525126 0.14833733
#  0.19007725 0.03698702 0.3676595  0.14087568]
# 하위 20퍼센트의 인덱스: [0 2]