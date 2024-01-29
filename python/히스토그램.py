import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE    # anaconda에서 사이킷런 설치할 때 같이 설치됨    없다면  pip install imblearn


csv_path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

encoder = LabelEncoder()
ohe = OneHotEncoder()

columns_to_drop = ['대출등급']
x = train_csv.drop(columns=columns_to_drop)
y = train_csv['대출등급']

# 대출 목적
encoder.fit(x['대출목적'])
x['대출목적'] = encoder.transform(x['대출목적'])

log_transformed_x = np.log1p(x['대출목적'])
# scaler = StandardScaler()
# scaler.fit(x['대출목적'])
# x['대출목적'] = scaler.transform(x['대출목적'])

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.figure(figsize=(14, 10))
sns.histplot(data=x, x='대출목적', bins=30, kde=True)  # kde=True는 커널 밀도 추정을 함께 표시
plt.title('Histogram of 특정컬럼')
plt.xlabel('특정컬럼 값')
plt.ylabel('빈도')
plt.show()
#--------------------------------


# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.hist(x['대출목적'], bins=50, color='blue', alpha=0.7)
# plt.title('Original Data Histogram')

# plt.subplot(1, 2, 2)
# plt.hist(log_transformed_x, bins=50, color='green', alpha=0.7)
# plt.title('Log Transformed Data Histogram')
# plt.show()
#----------------------------------------

# 이중 분포
# np.random.seed(42)
# data_mode1 = np.random.normal(loc=0, scale=1, size=1000)
# data_mode2 = np.random.normal(loc=5, scale=1, size=1000)

# # 데이터를 합침 (이중 모드 분포)
# bimodal_data = np.concatenate([data_mode1, data_mode2])

# # 이중 모드 분포를 시각화
# plt.figure(figsize=(8, 6))
# sns.histplot(bimodal_data, bins=30, kde=True)
# plt.title('Bimodal Distribution')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()

#---------------------------
# 정규 분포
# 정규 분포를 따르는 데이터 생성

# np.random.seed(42)
# normal_data = np.random.normal(loc=0, scale=1, size=1000)

# # 히스토그램 시각화
# plt.figure(figsize=(8, 6))
# sns.histplot(normal_data, bins=30, kde=True)
# plt.title('Histogram of Normal Distribution')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()