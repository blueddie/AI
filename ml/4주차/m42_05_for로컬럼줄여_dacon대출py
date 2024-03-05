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


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8, stratify=y)

#2
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(model.__class__.__name__, "acc :", results)  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")

# DecisionTreeClassifier acc : 0.8339477646814476
# DecisionTreeClassifier feature importance
# [5.73117819e-02 3.38675310e-02 1.39370163e-02 5.57315860e-03
#  3.50382677e-02 3.15680291e-02 2.57065397e-02 9.43680251e-03
#  6.31824231e-03 4.10751106e-01 3.69867187e-01 3.44894646e-04
#  2.79443277e-04]
# RandomForestClassifier acc : 0.8007684718832754
# RandomForestClassifier feature importance
# [0.09972816 0.02734712 0.04419498 0.01704241 0.08287851 0.08987033
#  0.07087642 0.02500054 0.01657949 0.26396716 0.26106141 0.00052444
#  0.00092901]
# GradientBoostingClassifier acc : 0.7436523183965938
# GradientBoostingClassifier feature importance
# [1.97077023e-02 1.11533666e-01 3.32738553e-05 8.60284752e-04
#  1.60923227e-02 6.54754798e-03 1.58252196e-03 6.43346954e-03
#  9.56643003e-04 3.90803150e-01 4.45377458e-01 5.06630178e-05
#  2.12963305e-05]
# XGBClassifier acc : 0.8541461135053741
# XGBClassifier feature importance
# [0.04721492 0.39846307 0.01172765 0.01734597 0.03443046 0.01772315
#  0.0139598  0.03024483 0.01987921 0.19287258 0.19310457 0.01168424
#  0.01134964]