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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



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

n_features = len(np.unique(y))


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8, stratify=y)

#2 모델
model = RandomForestClassifier(random_state=777)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

for i in range(1, n_features):
    
    lda = LinearDiscriminantAnalysis(n_components=i)   
    x1 = lda.fit_transform(x_train, y_train)
    x2 = lda.transform(x_test)

    # #3.
    model.fit(x1, y_train)

    # #4.
    results = model.score(x2, y_test)
    print(x1.shape)
    print(f"model.score : {results}")
    evr = lda.explained_variance_ratio_
    evr_cumsum = np.cumsum(evr)
    print(evr_cumsum)

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

# (77035, 1)
# model.score : 0.24289942364608755
# [0.18053739]
# (77035, 2)
# model.score : 0.28210187444830986
# [0.18053755 0.27978294]
# (77035, 3)
# model.score : 0.3717742354223999
# [0.18053738 0.27978314 0.3674908 ]
# (77035, 4)
# model.score : 0.4009034737006075
# [0.18053753 0.27978317 0.36749082 0.45141749]
# (77035, 5)
# model.score : 0.4135209512435744
# [0.18053745 0.27978315 0.36749075 0.45141738 0.52947391]
# (77035, 6)
# model.score : 0.43605586998286516
# [0.18053745 0.27978306 0.36749071 0.45141736 0.52947394 0.60687428]
# (77035, 7)
# model.score : 0.43112311127265174
# [0.18053755 0.27978321 0.36749082 0.45141754 0.52947413 0.60687437
#  0.68224219]
# (77035, 8)
# model.score : 0.4286826938054935
# [0.18053738 0.27978315 0.36749089 0.45141761 0.52947409 0.60687434
#  0.68224218 0.75515357]
# (77035, 9)
# model.score : 0.4301365595306091
# [0.18053747 0.27978328 0.36749099 0.45141772 0.52947432 0.60687462
#  0.68224247 0.7551539  0.82182257]
# (77035, 10)
# model.score : 0.4319019679111065
# [0.18053742 0.2797831  0.36749081 0.4514175  0.52947401 0.60687426
#  0.68224219 0.75515362 0.8218222  0.88599649]
# (77035, 11)
# model.score : 0.4299288644270211
# [0.18053748 0.27978325 0.36749096 0.45141761 0.52947407 0.60687431
#  0.68224219 0.75515354 0.82182218 0.88599649 0.94347893]
# (77035, 12)
# model.score : 0.4464406251622618
# [0.18053748 0.27978325 0.36749096 0.45141761 0.52947407 0.60687431
#  0.68224219 0.75515354 0.82182218 0.88599649 0.94347893 0.97325268]
# (77035, 13)
# model.score : 0.6218391401422712
# [0.18053748 0.27978325 0.36749096 0.45141761 0.52947407 0.60687431        # 제일 좋음
#  0.68224219 0.75515354 0.82182218 0.88599649 0.94347893 0.97325268 
# 1.]

# LDA
# (77035, 1)
# model.score : 0.32551015109818787
# [0.96161115]
# (77035, 2)
# model.score : 0.4023054156498261
# [0.96161115 0.99294566]
# (77035, 3)
# model.score : 0.464302404070824
# [0.96161115 0.99294566 0.99901756]
# (77035, 4)
# model.score : 0.47691988161379095
# [0.96161115 0.99294566 0.99901756 0.99960934]
# (77035, 5)
# model.score : 0.5119684303442547
# [0.96161115 0.99294566 0.99901756 0.99960934 0.99986577]
# (77035, 6)
# model.score : 0.5195493016252142
# [0.96161115 0.99294566 0.99901756 0.99960934 0.99986577 1.        ]