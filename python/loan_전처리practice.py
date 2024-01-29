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

csv_path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

df = train_csv

numeric_columns = df.select_dtypes(include=['int', 'float']).columns

# 대출등급을 제외한 연속형 변수 사용
columns_to_check = numeric_columns[numeric_columns != "대출등급"]

# Z-score를 계산하여 이상치 확인
z_scores = stats.zscore(df[columns_to_check])

# 임계값 설정 (보통 2 또는 3을 사용)
threshold = 3

# Z-score가 임계값을 넘어서는 데이터 포인트를 이상치로 간주
outliers = (z_scores > threshold).any(axis=1)

# 이상치 제거
df_no_outliers = df[~outliers]

# print("이상치 제거 전:", df.shape)

# # 이상치 제거 후의 행과 열 수
# print("이상치 제거 후:", df_no_outliers.shape)
# # 이상치 제거 전의 통계량
# print("이상치 제거 전 평균:\n", df[columns_to_check].mean())

# # 이상치 제거 후의 통계량
# print("이상치 제거 후 평균:\n", df_no_outliers[columns_to_check].mean())

xy = df_no_outliers

unknown_replacement = xy['근로기간'].mode()[0]
xy.loc[xy['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement

# xy['근로기간'] = xy['근로기간'].str.slice(0, 2)
# xy['근로기간'] = xy['근로기간'].str.strip()
# xy['근로기간'] = xy['근로기간'].replace({'<' : 0, '<1' :0})
xy.loc[xy['근로기간'] == '<1 year', '근로기간'] = '< 1 year'
xy.loc[xy['근로기간'] == '3', '근로기간'] = '3 years'
xy.loc[xy['근로기간'] == '10+years', '근로기간'] = '10+ years'
xy.loc[xy['근로기간'] == '1 years', '근로기간'] = '1 year'

encoder = LabelEncoder()
encoder.fit(xy['근로기간'])
xy.loc[:, '근로기간'] = encoder.transform(xy['근로기간'])

# print(pd.value_counts(xy['근로기간']))

# 대출 기간
encoder = LabelEncoder()
encoder.fit(xy['대출기간'])
xy.loc[:, '대출기간'] = encoder.transform(xy['대출기간'])
# test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])

# 주택소유상태
xy.loc[xy['주택소유상태'] == 'ANY', '주택소유상태'] = 'MORTGAGE'

encoder.fit(xy['주택소유상태'])
xy.loc[:, '주택소유상태'] = encoder.transform(xy['주택소유상태'])
# test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])


#대출 목적
# test_csv['대출목적'] = test_csv['대출목적'].replace({'결혼' : '부채 통합'})
encoder.fit(xy['대출목적'])
xy.loc[:, '대출목적'] = encoder.transform(xy['대출목적'])

# print(xy.shape) #(90293, 14)

columns_to_drop = ['대출등급']
xy = xy.drop(columns=columns_to_drop)
# y = xy['대출등급']

xy = xy.astype(float)
# xy['대출금액'] = xy['대출금액'].values.reshape(-1, 1)


# print(xy.shape)
#---------------
'''
# 로그 변환을 적용할 컬럼 선택
column_to_transform = '대출금액'

# 선택한 컬럼에 로그 변환 적용
xy[column_to_transform + '_log'] = np.log1p(xy[column_to_transform])

# 변환 전과 후의 히스토그램 비교
plt.figure(figsize=(12, 6))

# 원본 데이터의 히스토그램
plt.subplot(1, 2, 1)
sns.histplot(xy[column_to_transform], kde=True, color='blue')
plt.title('Original Histogram')

# 로그 변환 후의 히스토그램
plt.subplot(1, 2, 2)
sns.histplot(xy[column_to_transform + '_log'], kde=True, color='red')
plt.title('Log Transformed Histogram')

plt.show()

# 변환 전과 후의 기술 통계량 비교
original_stats = xy[column_to_transform].describe()
transformed_stats = xy[column_to_transform + '_log'].describe()

print("Original Statistics:")
print(original_stats)

print("\nLog Transformed Statistics:")
print(transformed_stats)


#   대출금액 : 스탠다드

'''


# #----------------
# plt.rcParams['font.family'] = 'Malgun Gothic' 
# plt.figure(figsize=(14, 10))
# sns.histplot(data=xy, x='대출금액', bins=30, kde=True)  # kde=True는 커널 밀도 추정을 함께 표시
# plt.title('Histogram of 특정컬럼')
# plt.xlabel('특정컬럼 값')
# plt.ylabel('빈도')
# plt.show()

#------------------
# 정규분포에 가까운 경우:

# 변환 없이 그대로 사용해도 될 수 있습니다.
# 일반적인 통계 분석이나 선형 모델에 적용할 때 좋은 결과를 낼 수 있습니다.
# 오른쪽으로 긴 꼬리를 가지고 있는 경우:

# 로그 변환 또는 Box-Cox 변환이 유용할 수 있습니다.
# 긴 꼬리를 줄여 정규분포에 가깝게 만들어줍니다.
# 왼쪽으로 긴 꼬리를 가지고 있는 경우:

# 제곱근 변환 등이 유용할 수 있습니다.
# 긴 꼬리를 줄여 정규분포에 가깝게 만들어줍니다.
# 분포가 일정 범위 내에 집중되어 있지 않은 경우:

# 특정 범위로 잘라내거나, 분포를 조정하는 방법을 고려할 수 있습니다.

#_______________   

#2 모델

