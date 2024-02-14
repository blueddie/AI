from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE    # anacon
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

csv_path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

xy = train_csv
#################################

unknown_replacement = xy['근로기간'].mode()[0]
xy.loc[xy['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement
test_csv.loc[test_csv['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement

xy.loc[xy['근로기간'] == '<1 year', '근로기간'] = '< 1 year'
xy.loc[xy['근로기간'] == '3', '근로기간'] = '3 years'
xy.loc[xy['근로기간'] == '10+years', '근로기간'] = '10+ years'
xy.loc[xy['근로기간'] == '1 years', '근로기간'] = '1 year'

test_csv.loc[test_csv['근로기간'] == '<1 year', '근로기간'] = '< 1 year'
test_csv.loc[test_csv['근로기간'] == '3', '근로기간'] = '3 years'
test_csv.loc[test_csv['근로기간'] == '10+years', '근로기간'] = '10+ years'
test_csv.loc[test_csv['근로기간'] == '1 years', '근로기간'] = '1 year'


encoder = LabelEncoder()
encoder.fit(xy['근로기간'])
xy['근로기간'] = encoder.transform(xy['근로기간'])
test_csv['근로기간'] = encoder.transform(test_csv['근로기간'])
# print(np.unique(test_csv['근로기간'], return_counts=True))
# print(pd.value_counts(xy['근로기간']))

# 대출 기간
encoder = LabelEncoder()
encoder.fit(xy['대출기간'])
xy['대출기간'] = encoder.transform(xy['대출기간'])
test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])

# 주택소유상태
xy.loc[xy['주택소유상태'] == 'ANY', '주택소유상태'] = 'MORTGAGE'

encoder.fit(xy['주택소유상태'])
xy['주택소유상태'] = encoder.transform(xy['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])
# test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])
# print(np.unique(xy['주택소유상태']))
# print(np.unique(test_csv['주택소유상태']))


#대출 목적
# print(np.unique(xy['대출목적']))
# print(np.unique(test_csv['대출목적']))
test_csv.loc[test_csv['대출목적'] == '결혼', '대출목적'] = '부채 통합'


encoder.fit(xy['대출목적'])
xy['대출목적'] = encoder.transform(xy['대출목적'])
test_csv['대출목적'] = encoder.transform(test_csv['대출목적'])
# print(np.unique(xy['대출목적']))
# print(np.unique(test_csv['대출목적']))
# print(xy.shape) #(90293, 14)

columns_to_drop = ['대출등급']
x = xy.drop(columns=columns_to_drop)

# print(x.shape)
x = x.astype(np.float32)

print(x.info())
y = xy['대출등급']

encoder = LabelEncoder()
y = encoder.fit_transform(y)
# 하위 20퍼센트의 인덱스: [ 2 11 12]
# x = x.drop(x.columns[12], axis=1)
# print(x.shape)
to_delete = [2,11,12]
for idx in sorted(to_delete, reverse=True):
    x = x.drop(x.columns[idx], axis=1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8, stratify=y)

#2
models = [DecisionTreeClassifier(random_state=777), RandomForestClassifier(random_state=777), GradientBoostingClassifier(random_state=777), XGBClassifier(random_state=777)]

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

# DecisionTreeClassifier acc : 0.8362843345968118
# DecisionTreeClassifier feature importance
# [5.77282321e-02 3.38675310e-02 1.29725923e-02 5.91411066e-03
#  3.58055492e-02 3.12392879e-02 2.52014167e-02 9.32766291e-03
#  6.37297473e-03 4.11495670e-01 3.69485178e-01 2.99448437e-04
#  2.90346572e-04]
# 하위 20퍼센트의 인덱스: [ 3 11 12]

# RandomForestClassifier acc : 0.8030531180227426
# RandomForestClassifier feature importance
# [0.09978064 0.0272578  0.04437829 0.01769875 0.08295319 0.09041176
#  0.07160005 0.02551091 0.0163615  0.26378679 0.2587955  0.00050959
#  0.00095523]
# 하위 20퍼센트의 인덱스: [ 8 11 12]

# GradientBoostingClassifier acc : 0.7436523183965938
# GradientBoostingClassifier feature importance
# [1.97021038e-02 1.11533666e-01 3.63457587e-05 8.59482682e-04
#  1.60912653e-02 6.55425187e-03 1.57842892e-03 6.43345479e-03
#  9.47860529e-04 3.90793501e-01 4.45387001e-01 5.74906921e-05
#  2.51476288e-05]
# 하위 20퍼센트의 인덱스: [ 2 11 12]

# XGBClassifier acc : 0.8541461135053741
# XGBClassifier feature importance
# [0.04721492 0.39846307 0.01172765 0.01734597 0.03443046 0.01772315
#  0.0139598  0.03024483 0.01987921 0.19287258 0.19310457 0.01168424
#  0.01134964]
# 하위 20퍼센트의 인덱스: [ 2 11 12]

#===========================================================================
# DecisionTreeClassifier acc : 0.8382574380808973
# DecisionTreeClassifier feature importance
# [0.06202079 0.03386753 0.00671123 0.03637053 0.03546835 0.02616259
#  0.00987799 0.00633895 0.41232791 0.37085413]
# 하위 20퍼센트의 인덱스: [2 7]

# RandomForestClassifier acc : 0.8392439898229399
# RandomForestClassifier feature importance
# [0.10124246 0.02792907 0.01580007 0.07497773 0.08106923 0.06347302
#  0.02238523 0.01497245 0.30370414 0.29444661]
# 하위 20퍼센트의 인덱스: [2 7] 

# GradientBoostingClassifier acc : 0.7435484708447998
# GradientBoostingClassifier feature importance
# [0.01996105 0.11004592 0.00091457 0.01597364 0.00648257 0.00147115
#  0.00627595 0.00103226 0.3931736  0.44466929]
# 하위 20퍼센트의 인덱스: [2 7]

# XGBClassifier acc : 0.8546134274884469
# XGBClassifier feature importance
# [0.04770174 0.4247708  0.01822925 0.03444969 0.01815601 0.01402147
#  0.02816641 0.01981337 0.19447951 0.20021173]
# 하위 20퍼센트의 인덱스: [4 5]