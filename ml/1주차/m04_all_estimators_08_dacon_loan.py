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
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

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

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8, stratify=y)

#2
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgoritms : ', allAlgorithms)
print('모델의 갯수',len(allAlgorithms)) #모델의 갯수 41

for name, algorithm in allAlgorithms:
    
    try:
        #2 모델
        model = algorithm()
        #3 훈련
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 : ', acc)
    except Exception as e:
        print(name , '에러 발생', e)
        continue